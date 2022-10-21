import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np





def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        gt = list(np.load('gt-colon.npy'))
        start_ind = 0
        gt_abn = []
        embedding_overall = torch.zeros(0)
        for i, (input, filename) in enumerate(dataloader):
            input = input.to(device)
            filename = filename[0].split('.npy')[0].split('/')[-1]

            input = input.squeeze(2)
            pred_temp = torch.zeros(0)
            len_num_seg = input.shape[1]
            # print(len_num_seg)


            embedding_temp  = torch.zeros(0)

            for j in range(input.shape[1]//32+1):
                start_idx = j * 32
                end_idx = (j + 1)*32
                # print(start_idx)
                # print(end_idx)

                input_tmp = input[:, start_idx:end_idx, :]
                if input_tmp.shape[1] < 32:
                    for last in range((32-input_tmp.shape[1])):
                        input_tmp = torch.cat((input_tmp, input[:, -1, :].unsqueeze(1)), dim=1)
                x, cls_tokens, cls_prob,  scores, _, embeddings = model(input_tmp)
                embeddings = embeddings.squeeze(0)

                logits = torch.squeeze(scores, 2)
                logits = torch.mean(logits, 0) 
                sig = logits
                pred_temp = torch.cat((pred_temp, sig))
                # embedding_temp = torch.cat((embedding_temp, embeddings), dim=0)

            # logits = torch.squeeze(scores, 2)
            # logits = torch.mean(logits, 0)
            # sig = logits
            # pred = torch.cat((pred, sig))


            # print(start_ind)

            pred = torch.cat((pred, pred_temp[:len_num_seg]))


            # pred_plot = pred_temp[:len_num_seg].cpu().detach().numpy()
            # pred_plot = np.repeat(np.array(pred_plot), 16)



            # axes = plt.gca()
            # axes.set_ylim([-0.05, 1.05])
            
            # frames = np.arange(0, pred_plot.shape[0])


            # plt.plot(frames, pred_plot, color='orange',  linewidth=3)
            # plt.xlabel('Frame Number', fontsize=15)
            # plt.ylabel('Anomaly Score', fontsize=15)
            # plt.grid(False)
            # axes.xaxis.set_tick_params(labelsize=15)
            # axes.yaxis.set_tick_params(labelsize=15)

            # plt.savefig('plot_img/pred_' + str(filename) + '.png')
            # plt.close()

            # end_ind = start_ind+pred_plot.shape[0]
            
            # gt_plot = gt[start_ind:end_ind]
            # start_ind = end_ind
            
 

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        
        rec_auc = auc(fpr, tpr)
        
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)

        print('auc : ' + str(rec_auc))
        print('AP : ' + str(pr_auc))
        print('pr_auc {}'.format(pr_auc))
        
        return rec_auc

