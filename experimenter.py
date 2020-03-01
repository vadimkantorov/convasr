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
    --train-batch-size 256 --val-batch-size 256 \
    --scheduler MultiStepLR --decay-milestones 40000 \
    --lr 1e-2 \
    --optimizer NovoGrad \
    --train-data-path data/domain_splits/{train_set}.csv \
    --val-data-path data/domain_splits/mixed_val.csv data/domain_splits/clean_val.csv  kontur_calls_micro/kontur_calls_micro.csv kontur_calls_micro/kontur_calls_micro.0.csv kontur_calls_micro/kontur_calls_micro.1.csv data/valset_by_rec22122019.0.csv data/valset_by_rec22122019.1.csv data/valset_by_rec22122019.csv \
    --analyze kontur_calls_micro.csv \
    --val-iteration-interval 2500 \
    --fp16 O2 \
    --experiment-name domain_comp_{train_set} \
    --epochs 45 
"""

def get_process_to_gpu_mapping():
    return {
        '1': '0',
        '2': '1',
        '3': '2',
        '4': '3',
    }


def run_task(worker_id, args):
    bashcmd = bash_template.format(train_set=args)
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

    for v in ['radio_normalized_train',
	'books_normalized_train',
	'clean_train',
	'books_radio_normalized_train',
	'books_youtube_normalized_train',  
	'youtube_normalized_train',
	'books_youtube_radio_normalized_train',
	'youtube_radio_normalized_train']:
        queue.put(v)

    pool = Pool(4, worker_loop, (queue,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
