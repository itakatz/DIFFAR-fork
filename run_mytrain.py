import hydra
import pickle
from learner import mytrain


@hydra.main(config_path="conf", config_name="conf_ssynth_spec_loss", version_base = None)
def main(params):
    mytrain(0, 0, 0, params)

'''
[Itamar Katz]
my simple version of runing training, for getting familiar with the model and training
it just skips the hydra stuff (I debugged the hydra-run and saved "params" to disk; normally it is read by hydra from "conf/conf.yaml")
'''

if __name__ == '__main__':
    if False:
        params = pickle.load(open('/home/mlspeech/itamark/git_repos/DIFFAR/runs/DiffAR_200/outputs/exp_/params_for_mytrain.pickle','rb'))
        params.residual_layers = 32 # avoid GPU out-of-mem error with 36 layers
        #--- add value not in conf yaml (TOOD)
        from omegaconf import open_dict
        with open_dict(params):
            params.mask_loss_using_overlap = False

        #--- run with no distributed learning
        mytrain(0, 0, 0, params)

    main()
