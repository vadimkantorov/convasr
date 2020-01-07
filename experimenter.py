import time
from multiprocessing import Queue, Pool
import multiprocessing as mp
import subprocess

bash_template = "python3 run.py --arg {arg}"

bash_template = """
/miniconda/bin/python3 train.py \
	--githttp https://github.com/vadimkantorov/convasr/commit/%h \
	--verbose --lang ru \
	--model JasperNetBig \
	--train-batch-size 256 --val-batch-size 256 \
	--scheduler MultiStepLR --decay-milestones 20000 35000 \
	--lr 1e-2 \
	--optimizer NovoGrad \
	--train-data-path data/splits/mixed/mixed_with_radio_train_{set_number}.csv \
	--val-data-path data/splits/radio_val.csv data/mixed_val.csv data/clean_val.csv kontur_calls_micro/kontur_calls_micro.csv kontur_calls_micro/kontur_calls_micro.0.csv kontur_calls_micro/kontur_calls_micro.1.csv \
	--analyze kontur_calls_micro.csv \
	--val-iteration-interval 2500 \
	--experiment-id radio_domainset_mixed_{set_number} \
	--fp16 O2 \
	--epochs 8
"""

def get_process_to_gpu_mapping():
    return {
        '1': '0',
        '2': '1',
        '3': '2',
        '4': '3',
    }


def run_task(worker_id, args):
    bashcmd = bash_template.format(set_number=args)
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

    for v in [1, 2, 3, 5, 10, 11, 14]:
        queue.put(v)

    pool = Pool(1, worker_loop, (queue,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

