import time
from multiprocessing import Queue, Pool
import multiprocessing as mp
import subprocess


bash_template = """
/miniconda/bin/python3 train.py \
	--githttp https://github.com/vadimkantorov/convasr/commit/%h \
	--verbose --lang ru \
	--model JasperNetBig \
	--train-batch-size 256 --val-batch-size 256 \
	--scheduler MultiStepLR --decay-milestones 145000 155000 \
	--lr 1e-3 \
	--optimizer NovoGrad \
	--train-data-path data/splits/radio_train_{set_number}.csv \
	--val-data-path data/splits/radio_val.csv data/mixed_val.csv data/clean_val.csv kontur_calls_micro/kontur_calls_micro.csv kontur_calls_micro/kontur_calls_micro.0.csv kontur_calls_micro/kontur_calls_micro.1.csv \
	--analyze kontur_calls_micro.csv \
	--val-iteration-interval 2500 \
	--experiment-name radio_finetune_1423_{set_number} \
	--checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-3_wd1e-3_bs256____pretrain_11/checkpoint_epoch09_iter0034156.pt \
	--fp16 O2 \
	--epochs {epochs}
"""
# data/experiments/JasperNetBig_NovoGrad_lr1e-3_wd1e-3_bs256____pretrain_11/checkpoint_epoch09_iter0034156.pt
#  data/experiments/radio_domainset_0/checkpoint_epoch04_iter0028600.pt 
def get_process_to_gpu_mapping():
    return {
        '1': '0',
        '2': '1',
        '3': '2',
        '4': '3',
    }


def run_task(worker_id, args):
    epochs = args[1]
    set_number = args[0]

    bashcmd = bash_template.format(set_number=set_number, epochs=epochs)
    print(bashcmd)

    with open(f'data/experiments/experiment_{set_number}_output.txt', 'a+') as out:
        with open(f'data/experiments/subprocess_{set_number}_exit_codes.txt', 'a+') as experiment_results:
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

    for v in [(1, 30), (2, 20), (3, 15), (5, 12), (10, 10), (11, 7), (14, 5)]:
        queue.put(v)

    pool = Pool(2, worker_loop, (queue,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

