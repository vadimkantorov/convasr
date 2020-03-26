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
    --train-batch-size 32 --val-batch-size 128 \
    --lr 5e-4 \
    --optimizer NovoGrad \
    --train-data-path {train_set} \
    --val-data-path data/domain_splits/mixed_val.csv data/domain_splits/clean_val.csv {val_set} \
    --analyze kontur_calls_micro.csv \
    --val-iteration-interval 2500 \
    --fp16 O2 \
    --checkpoint {cp} \
    --experiment-name finetune_domain_comb_2_{i} \
    --epochs {epoch} 
"""

def get_process_to_gpu_mapping():
    return {
        '1': '0',
        '2': '1',
        '3': '2',
        '4': '3',
    }


def run_task(worker_id, args):
    (i, epoch,  train_set, val_set, cp) = args
    print(type(worker_id), worker_id)
    print(train_set)
    print(val_set)
    print(cp)
    print(epoch)
    epoch = int(cp.split('_')[-2].replace('epoch', ''))
    print(epoch) 
    bashcmd = bash_template.format(i=i, epoch=epoch+20, train_set='data/kfold_splits/' + train_set, val_set =' '.join(['data/kfold_splits/' + vs for vs in val_set.split(' ')]), cp=cp)
    print(bashcmd)
    with open(f'data/experiments/experiment_{i}_output.txt', 'a+') as out:
        with open(f'data/experiments/subprocess_{i}_exit_codes.txt', 'a+') as experiment_results:
            exit_code = subprocess.call(bashcmd.split(),
                                        stdout=out,
                                        stderr=out,
                                        env={"CUDA_VISIBLE_DEVICES": f"{get_process_to_gpu_mapping()[worker_id]}"}, )

            experiment_results.write(f"{i}\t{exit_code}\n")


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

    for v in [
	(6, 38, 'trainset_kfold_16032020_fold_1.csv', 'valset_kfold_16032020_fold_1.csv valset_kfold_16032020.1_fold_1.csv valset_kfold_16032020.0_fold_1.csv', 'data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comb_max_qual2_books_radio_train/checkpoint_epoch46_iter0169924.pt'),
	(7, 38, 'trainset_kfold_16032020_fold_1.csv', 'valset_kfold_16032020_fold_1.csv valset_kfold_16032020.1_fold_1.csv valset_kfold_16032020.0_fold_1.csv', 'data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comb_max_qual2_books_train/checkpoint_epoch74_iter0170000.pt'),
	(8, 38, 'trainset_kfold_16032020_fold_1.csv', 'valset_kfold_16032020_fold_1.csv valset_kfold_16032020.1_fold_1.csv valset_kfold_16032020.0_fold_1.csv', 'data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____domain_comb_max_qual2_radio_train/checkpoint_epoch113_iter0170000.pt'),	
	]:
        queue.put(v)

    pool = Pool(3, worker_loop, (queue,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
