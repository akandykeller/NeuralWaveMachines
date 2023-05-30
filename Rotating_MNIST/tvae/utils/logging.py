import os

# def get_dirs():
#     cwd = os.path.dirname(os.path.realpath(__file__))

#     local_savedir = cwd
#     local_datadir = cwd
#     local_wandbdir = cwd

#     return local_savedir, local_datadir, local_wandbdir


# def configure_logging(config, name, model):
#     if config['wandb_on']:
#         import wandb

#         wandb.init(name=name,
#                    project='YOUR_PROJECT_NAME', 
#                    entity='YOUR_ENTITY_NAME', 
#                    dir=config['wandb_dir'],
#                    config=config)
#         wandb.watch(model)

#         def log(key, val):
#             print(f"{key}: {val}")
#             wandb.log({key: val})

#         checkpoint_path = os.path.join(wandb.run.dir, 'checkpoint.tar')
#     else:
#         def log(key, val):
#             print(f"{key}: {val}")
#         checkpoint_path = './checkpoint.tar'

#     return log, checkpoint_path


def get_dirs():
    cwd = os.path.dirname(os.path.realpath(__file__))

    remote_savedir = '/var/scratch/takeller/TECA/'
    remote_datadir = '/home/takeller/TECA/tvae/datasets/'
    remote_wandbdir = '/var/scratch/takeller/TECA/'

    local_savedir = '/home/akeller/repo/TVAE/tvae/results/'
    local_datadir = '/home/akeller/repo/TVAE/tvae/datasets/'
    local_wandbdir = '/home/akeller/repo/TVAE/'

    if 'takeller' in cwd:
        return remote_savedir, remote_datadir, remote_wandbdir
    else:
        return local_savedir, local_datadir, local_wandbdir


def configure_logging(config, name, model=None):
    if config['wandb_on']:
        import wandb

        wandb.init(name=name,
                   project='TECA', 
                   entity='akandykeller', 
                   dir=config['wandb_dir'],
                   config=config)
        if model is not None:
            wandb.watch(model)

        def log(key, val):
            print(f"{key}: {val}")
            wandb.log({key: val})

        checkpoint_path = os.path.join(wandb.run.dir, 'checkpoint.tar')
    else:
        def log(key, val):
            print(f"{key}: {val}")
        checkpoint_path = './checkpoint.tar'

    return log, checkpoint_path