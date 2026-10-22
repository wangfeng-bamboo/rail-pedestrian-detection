import matplotlib.pyplot as plt
# 解决中文方框乱码核心配置
plt.rcParams["font.family"] = ["Microsoft YaHei"]  # Windows微软雅黑
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

fig, ax = plt.subplots(figsize=(16,9),facecolor='white')
ax.set_xlim(0,10)
ax.set_ylim(0,6)
ax.axis('off')

# 标题
plt.text(5,5.6,"网络整体架构设计（Darknet残差主干）",fontsize=26,ha='center',color='#1a3399',weight='bold')

# 中间大圆
circle = plt.Circle((5,3),1.2,color='#253490')
ax.add_patch(circle)
plt.text(5,3,"SimpleDet\nNet",fontsize=22,ha='center',color='white',weight='bold')

# 四段中文说明
plt.text(2,4.2,"主干Backbone: Darknet卷积+残差\n模块，32倍下采样输出13×13特征图，\n缓解梯度消失",fontsize=13)
plt.text(7.2,4.2,"检测头Head: 两层卷积+1×1输出层，\n输出网格预测框、置信度、类别得分",fontsize=13)
plt.text(2,1.6,"多任务联合损失: 坐标损失+置信度损\n失+分类损失协同优化",fontsize=13)
plt.text(7.2,1.6,"后处理: NMS非极大值抑制过滤冗\n余检测框",fontsize=13)

# 左侧空白框
rect = plt.Rectangle((0.2,1.2),2.2,3.6,fill=False,lw=1)


plt.tight_layout()
plt.savefig("网络架构图.png",dpi=300)
plt.show()