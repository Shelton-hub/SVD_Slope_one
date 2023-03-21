import numpy as np
import random
import time
def getData():  # 获取全训练集,SVD训练集和测试集的函数
    import re
    f = open("ml-100k/u1.base", 'r')
    lines = f.readlines()
    f.close()
    data = []
    SVDdata=[]
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
            j=int(list[0])
            i=int(list[1])
            #挑选偶数号用户，奇数号项目
            if j%2==0 and i%2==1:
                SVDdata.append([int(i) for i in list[:3]])
    fulltrain_data = data
    f = open("ml-100k/u1.test", 'r')
    lines = f.readlines()
    f.close()
    data = []
    for line in lines:
        list = re.split('\t|\n', line)
        if int(list[2]) != 0:
            data.append([int(i) for i in list[:3]])
    test_data = data
    '''
    print("从文件读取训练集")
    for j in range(1,10):
        print("来自全训练数据集:",fulltrain_data[j])
        print("来自SVD训练数据集:",SVDdata[j])
    '''
    return fulltrain_data,SVDdata,test_data

class SVD:
    def __init__(self, mat, K=20):
        self.mat = np.array(mat)#数组的每个元素为一个有三个元素的数组
        self.K = K
        self.bi = {}#字典，item bias
        self.bu = {}#user bias
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2])#评分的均值
        print("mat.shape:",self.mat.shape)
        for i in range(self.mat.shape[0]):#shape返回（评分数量，评分信息的维度(如3个元素的数组，返回3)）
            uid = self.mat[i, 0]#第i个数组（代表第i个评分信息）的第0个元素，是评分用户
            iid = self.mat[i, 1]
            #print("uid,iid:",uid,iid)
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)#最初的偏置都设为0
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))#随机生成关于iid一个k维向量,加入qi字典
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))#随机生成关于uid一个k维向量
        print("bi:",len(self.bi))
        print("bu:",len(self.bu))
        #print("pu:",len(self.pu))
    def predict(self, uid, iid):  # 预测评分的函数
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0

        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))

        rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * self.pu[uid])  # biasSVD预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def train(self, steps=10, gamma=0.04, Lambda=0.15):  # 训练函数，step为迭代次数。
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])  # 随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                eui = rating - self.predict(uid, iid)
                rmse += eui ** 2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                tmp = self.qi[iid]
                self.qi[iid] += gamma * (eui * self.pu[uid] - Lambda * self.qi[iid])
                self.pu[uid] += gamma * (eui * tmp - Lambda * self.pu[uid])
            gamma = 0.93 * gamma# gamma以0.93的学习率递减
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))

    def test(self, test_data):
        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))

def getData2(fulltrain_data,a):  # 输入完整的训练集,a是传入的SVD类,获取第二个训练集的函数,返回一个评分字典
    #print("填充之前个数",len(train_data))
    #print("第二次读取数据集")
    #for j in range(1,10):
    #   print(train_data[j])
    print("开始获得中间数据集")
    data = {}
    #先把全训练集的数据补全,维度变回原来的用户，项目数量
    for record in fulltrain_data:
        data.setdefault(record[0], {})#list[0]是用户id,list[1]项目id,list[2]评分
        data[record[0]].setdefault(record[1],int(record[2]))
    #sum = 0
    #for i in data.keys():
    #   sum += len(data[i])
    #print("data字典构造完毕,填充之前个数", sum)
    #print("第二次读取数据集")
    #print(data[1])

    #把用了一半用户，一半项目训练的SVD的数据补全
    for (user,ubias) in a.bu.items():
        for(item,ibias) in a.bi.items():
                if user in data.keys():
                    if item in (data[user]).keys():
                        continue
                    else:
                        data[user].setdefault(item, a.predict(user, item))
                else:
                    data.setdefault(user,{})
                    data[user].setdefault(item,a.predict( user, item))

    #train_data = data return train_data


    sum=0
    for i in data.keys():
        sum+=len(data[i])
    print("填充之后个数",sum)
    #print(data)
    print("成功获得中间数据集")
    #print(data[1])
    return data

def computeDeviations(data):#data是个字典,生成slopeone的两个重要字典信息，并写在文件devi.txt中,避免每次都计算
    print("开始计算项目之间流行程度差字典和频率字典")
    deviations={}#流行程度差
    frequency={}#权重,同时评论两个项目的用户数目
    for bands in data.values():
        for (artist,rating) in bands.items():
            deviations.setdefault(artist,{})
            frequency.setdefault(artist,{})
            for (artist2,rating2) in bands.items():
                if artist2 != artist:
                    frequency[artist].setdefault(artist2, 0.0)
                    frequency[artist][artist2]+=1#同时评论这两个item的权重增加

                    deviations[artist].setdefault(artist2, 1)
                    deviations[artist][artist2]*= (rating/rating2)#记录流行程度差和
    f = open('ml-100k/devimul.txt', 'w')
    for (item,ratings) in deviations.items():
        for item2 in ratings:
            deviations[item][item2] = pow(deviations[item][item2], 1 / frequency[item][item2])  # 开方同时评价了两个项目的用户个数
            f.write(str(item)+'\t'+str(item2)+'\t'+ str(deviations[item][item2])+'\t'+str(frequency[item][item2])+'\n')
    f.close()
    print("成功返回项目流行度字典")
    return deviations,frequency

def getdevimul():  # 加载slope_one算法的流行度，频率表到内存
    import re
    f = open("ml-100k/devimul.txt", 'r')
    lines = f.readlines()
    f.close()
    deviations = {}
    frequency={}
    for line in lines:
        list = re.split('\t', line)
        item1 = int(list[0])
        item2 = int(list[1])
        diff=float(list[2])
        freq=float(list[3])
        deviations.setdefault(item1, {})
        frequency.setdefault(item1, {})
        frequency[item1].setdefault(item2, 0.0)
        frequency[item1][item2] =  freq
        deviations[item1].setdefault(item2, 0.0)
        deviations[item1][item2] = diff # 记录流行程度差和
    print("加载流行度表完毕")
    return deviations,frequency

def prediction(userid,itemid,data,deviations,frequency):#预测用户userid给itemid的评分,data是已评分字典，返回评分值
    user=data[userid]# 这个user实际上是这个用户评分过的各个(item,rating)pairs
    #print("user:",user)
    #print("user.items()",user.items())
    #print("itemid:",itemid)
    if itemid not in user:#如果用户user没有预测项目itemid,则给它预测分数
                summation = 0.0
                freq = 0.0
                recommend=0
                for (userItem, userRating) in user.items():
                    #print("UserItem:",userItem,"UserRating:",userRating)
                    if itemid in deviations.keys()and itemid in frequency.keys()and userItem in deviations[itemid].keys() and userItem in frequency[itemid].keys():
                        #print("deviations:", deviations[itemid][userItem], "userRating", userRating, "frequency",frequency[itemid][userItem])
                        summation += (deviations[itemid][userItem] * userRating) * frequency[itemid][userItem]
                        freq += frequency[itemid][userItem]
                        #print("summation:", summation, "sumfreq:", freq)

                if(freq!=0):
                    recommend = summation / freq
                else:
                    recommend=2.5
                if recommend > 5:
                    recommend = 5
                if recommend < 1:
                    recommend = 1
                return recommend
    else:
        return  data[userid][itemid]


#SVD+slope_one 最终的误差测试
def RMSEtest(data,test_data):#这个test_data是一个列表列表,data是字典
    deviations, frequency = getdevimul()
    test_data = np.array(test_data)
    print('test data size', test_data.shape)
    rmse = 0.0
    for i in range(test_data.shape[0]):
        uid = int(test_data[i, 0])
        iid = int(test_data[i, 1])
        rating = float(test_data[i, 2])
        #print("uid",uid)
        eui = rating - prediction(uid,iid,data,deviations,frequency)
        #print("计算得评分差")
        rmse += eui ** 2
    print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))

start=time.time()
fulltrain_data, SVDdata, test_data = getData()
a = SVD(SVDdata)
print("开始SVD")
a.train()
train_data2=getData2(fulltrain_data,a)
print("开始slope one")
RMSEtest(train_data2,test_data)
end=time.time()
print("运行时间:",end-start)



