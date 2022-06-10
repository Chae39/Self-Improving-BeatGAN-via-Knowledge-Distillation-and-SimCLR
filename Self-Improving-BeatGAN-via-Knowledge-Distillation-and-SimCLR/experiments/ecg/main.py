import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from options import Options

from data import load_data

# from dcgan import DCGAN as myModel


device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")


opt = Options().parse()
print(opt)
dataloader=load_data(opt)
print("load data success!!!")


if opt.model == "beatgan": 
    from model import BeatGAN_t as MyModel_t ## teacher model
    model_t=MyModel_t(opt,dataloader,device)

    if not opt.istest:
        print("################  Train  ##################")
        model_t.train()

    else:
        print("################  Eval  ##################")
        model_t.load()
        model_t.test_type()

elif opt.model=="beatgan_s":
    from model import BeatGAN_s as MyModel_s ## student model
    model_s=MyModel_s(opt,dataloader,device)

    if not opt.istest:
        print("################  Train  ##################")
        model_s.train()

    else:
        print("################  Eval  ##################")
        model_s.load()
        model_s.test_type()
else:
    raise Exception("no this model :{}".format(opt.model))

"""
if not opt.istest:
    print("################  Train  ##################")
    model.train()
    
else:
    print("################  Eval  ##################")
    #model.load()
    #model.test_type()

    #model.test_time()
    # model.plotTestFig()
    # print("threshold:{}\tf1-score:{}\tauc:{}".format( th, f1, auc))

"""