import sys
import json

class Settings:
    available_settings = {

        # Description 

        "description" ,    
       "dataset",  
       "multi_gpus",  
       "manual_seed", 

       # paths

       "data_root",
      "save_root",
      "model_path",
      "anno_root",
      "compute_paths",
      "compute_tubes", 

      #backbone

       "arch",
      "model_type",
      "anchor_type",

      #sequence

      "seq_len",
      "test_seq_len",
      "min_seq_step",
      "max_seq_step",

      #model_parameters

      "head_layers",
      "num_feature_maps",
      "cls_head_time_size",
      "reg_head_time_size",
      "freeze_bn",
      "freeze_upto",

      #dataloading

      "batch_size",
      "test_batch_size",
      "num_workers",
      "train_subsets",
      "val_subsets",
      "test_subsets",
      "min_size",

      #optimizer

      "optimizer",
      "optim",
      "resume",
      "max_epochs" ,
      "learning_rate",
      "momentum",
      "milestones",
      "gamma",
      "weight_decay",

      #loss_thresholds

      "positive_threshold",
      "negative_threshold",

      #"evaluation"

      "eval_epochs",
      "val_step",
      "iou_thresh",
      "conf_thresh",
      "nms_thresh",
      "topk",
      "gen_conf_thresh",
      "gen_topk",
      "gen_nms",
      "classwise_nms",
      "joint_4m_marginals", 

      #path_parameters

        "paths_iouth",
      "paths_cost_type",
      "paths_jump_gap",
      "paths_min_len",
      "paths_minscore",

     # tube_parameters
      "tubes_alpha",
      "trim_method",
      "tubes_topk",
      "tubes_minlen",
      "tubes_eval_threshs",

      # logging
       "log_start",
      "log_step",
      "tensorboard",

    }

    def __init__(self):
        try: 
            if len(sys.argv) <= 1:
                raise IndexError("Configuration file path is missing in the command-line arguments.")

            settings_config = sys.argv[1]
            user_settings = json.load(open(settings_config, "r"))
            print(user_settings)

            for user_settings_key in user_settings:
                print("Setting Key:", user_settings_key)
                setattr(self, user_settings_key, user_settings[user_settings_key])

            list(getattr(self, attr, None) for attr in self.available_settings)

        except IndexError as exc: 
            print(" No config found. Exception: ", exc)
            exit(-1)

        except FileNotFoundError as exc:
            print("Congfig argument found, but there is no file. Exception: ", exc)
            exit(-1)

        except AttributeError as exc:
            print ("Wrong Setting Provided or missing in the config file. Exception: ", exc)
            exit(-1)
    
Settings()
