import time
from multiprocessing import Queue, Pool
import multiprocessing as mp
import subprocess

bash_template = "python3 run.py --arg {arg}"

bash_template = """
/miniconda/bin/python3  train.py \
    --githttp https://github.com/vadimkantorov/convasr/commit/%h \
    --verbose --lang ru \
    --model JasperNetBig \
    --train-batch-size 16 --val-batch-size 256 \
    --scheduler MultiStepLR --decay-milestones 100000 175000 \
    --lr 5e-3 \
    --optimizer NovoGrad \
    --train-data-path data/kfold_splits/trainset_kfold_05022020_fold_{fold}.csv \
    --val-data-path data/mixed_val.csv data/clean_val.csv data/kfold_splits/valset_kfold_05022020_fold_{fold}.csv data/kfold_splits/valset_kfold_05022020.0_fold_{fold}.csv  data/kfold_splits/valset_kfold_05022020.1_fold_{fold}.csv 
    --analyze kontur_calls_micro.csv \
    --val-iteration-interval 2500 \
    --fp16 O2 \
    --experiment-name finetune_kfold_{fold} \
    --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-3_wd1e-3_bs512____long-train_lr1e-4_checkpoint_epoch48_iter0173280.pt \
    --epochs 70 
"""

def get_process_to_gpu_mapping():
    return {
        '1': '0',
        '2': '1',
        '3': '2',
        '4': '3',
    }


def run_task(worker_id, args):
    bashcmd = bash_template.format(fold=args)
    print(bashcmd)

    with open(f'data/experiments/experiment_{args}_output.txt', 'a+') as out:
        with open(f'data/experiments/subprocess_{args}_exit_codes.txt', 'a+') as experiment_results:
            exit_code = subprocess.call(bashcmd.split(),
                                        stdout=out,
                                        stderr=out,
                                        env={"CUDA_VISIBLE_DEVICES": f"{get_process_to_gpu_mapping()[worker_id]}"}, )

            experiment_results.write(f"{args}\t{exit_code}\n")


def worker_loop(queue):
    while not queue.empty():
        task = queue.get()
        process = mp.current_process()
        worker_id = process.name.split('-')[1]

        print(f'worker id: {worker_id}, run new task: {task}')
        run_task(worker_id, task)

        time.sleep(30)
        if queue.empty():
            break


def main():
    queue = Queue()

    for v in [0]:
        queue.put(v)

    pool = Pool(1, worker_loop, (queue,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
