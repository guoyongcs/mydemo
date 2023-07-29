-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/opt/conda/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 59, in _wrap
    fn(i, *args)
  File "/workspace/algo/XFY_MQBench/application/imagenet_example/main.py", line 358, in main_worker
    os.makedirs(name=os.path.join(args.output_dir, 'ckpt',))
  File "/opt/conda/lib/python3.8/os.py", line 223, in makedirs
    mkdir(name, mode)
FileExistsError: [Errno 17] File exists: '/workspace/service/outputs/3670fe18-a244-4ceb-b098-f1a5b14af691/ckpt'
