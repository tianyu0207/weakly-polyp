from torch.utils.data import DataLoader
import torch
import os
from cvt_model_1d import get_model_transformer
from dataset import Dataset
from test_transformer import test
import option


torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':
    args = option.parser.parse_args()
    device = torch.device("cuda")  

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model =  get_model_transformer()
    for name, value in model.named_parameters():
        print(name)

    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('best_ap_ckpt.pkl').items()})
   
    auc = test(test_loader, model, args, device)

    print('The AUC of this ckpt is: ' + str(auc))

