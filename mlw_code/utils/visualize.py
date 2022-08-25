from sklearn import metrics
import matplotlib.pyplot as plt
import visdom
import time
import numpy as np

# python -m visdom.server
# http://localhost:8097


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（'loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don't ~~self.img('input_imgs', t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def roc_curve(values):
    '''
    Parameters:
    values: shape, [num_curves, 2, num_samples]. Two values (target, predicted_score) per sample per curve.
    '''
    fig = plt.figure()
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.plot([0, 1], [0, 1], color=[0.6, 0.6, 0.6], lw=0.8, linestyle='--')
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for idx, (targets, scores) in enumerate(values):
        fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
        auc = metrics.roc_auc_score(targets, scores, average=None)
        RightIndex = (tpr+(1-fpr)-1)
        index = np.where(RightIndex == max(RightIndex))

        tpr_val = tpr[index]
        fpr_val = fpr[index]
        thresholds_val = thresholds[index]  # 0.468
        print(thresholds_val)

        fpr = np.insert(fpr, 0, 0)
        tpr = np.insert(tpr, 0, 0)
        cohort_name = 'Training' if idx == 0 else 'Test'
        plt.plot(fpr, tpr, lw=2, label='%s cohort (AUC = %.3f)' % (cohort_name, auc))
        print(auc)

    plt.legend(loc=4, fontsize=12)
    plt.tight_layout()
    plt.show()
