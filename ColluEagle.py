import time
import operator
import datetime
import math
from scipy.stats import norm
import pandas as pd
import networkx as nx

GSMRF_HOME = "E:/DMKD submission/ColluEagle/"

FILE_META = "metadata"
FILE_PRIOR = 'ZIP_SpEagle_Review2Reviewer_Prior.txt' # use specified prior file
USE_PRIOR = True

DF_FROM = 200
DF_TO = 4200
TWSIZE = 4000  # time window in days
TW_DETECT_STEP = 4000
PRIOR_FRAUD = 0.2
SIGMA1 = 90
SIGMA2 = 3
MIN_SIM = 0.6
# node state transform probability matrix
# potential = [[math.log(0.85), math.log(0.15)], [math.log(0.15), math.log(0.85)]]  # for similar edge
PROB_TIME = norm.cdf(range(0, -260, -1), 0, SIGMA1)
PROB_RATE = norm.cdf(range(0, -5, -1), 0, SIGMA2)

fRunFile = GSMRF_HOME + 'RunResult_{}_{}(e4wij).txt'.format(
    FILE_PRIOR.replace('.txt', '') if USE_PRIOR else "NoPrior", MIN_SIM)
ResultFile = GSMRF_HOME + 'ColluEagle_{}_{}_{}_{}(e4wij)'.format(
    FILE_PRIOR.replace('.txt', '') if USE_PRIOR else 'NoPrior',
    str(SIGMA1), str(SIGMA2), str(MIN_SIM))


class TimeWindow:
    def __init__(self, begin, size, gvlist, rpdict):
        self.Graph = nx.Graph()
        self.twBegin = begin
        self.twSize = size

        self.gListReview = gvlist  # 全局评论列表指针
        self.twListV = []  # 当前窗口内评论的序号列表，指向gListReview, 序号从0开始
        self.reviewerGraph = {}  # reviewer graph in a time window, key = rid, value = [phi, prodmsg, belief,edgelist]

        self.twDictReviewer = {}  # 评论人所评论产品字典，key = rid, value = vid list
        self.twDictProduct = {}  # 产品对应的所有评论人字典, key = pid, value = vid list
        self.twDictRP = rpdict  # 评论字典，根据评论人编号和产品编号，查询评论, key = (rid,pid), value = a review index

    def LoadReviews(self):  # 生成当前时间窗对应的评论图
        for i, v in enumerate(self.gListReview):
            if self.twBegin + self.twSize > v[5] >= self.twBegin:  # V[5],日期
                rid = v[0]
                pid = v[1]
                label = v[3]
                if rid in self.twDictReviewer:
                    self.twDictReviewer[rid].append(pid)  # 添加该评论人的在本时间窗内的所有产品号，允许一个人对一个产品多次评论
                    if label == 1:
                        self.reviewerGraph[rid][4] = 1
                else:
                    self.twDictReviewer[rid] = [pid]  # pidlist
                    prior = [math.log(1 - PRIOR_FRAUD), math.log(PRIOR_FRAUD)]  # Default None Prior
                    self.reviewerGraph[rid] = [prior, [0, 0], [0, 0], [],
                                               label]  # node prior, prod message, belief, neighbor, label
                if pid in self.twDictProduct:
                    self.twDictProduct[pid].append(rid)
                else:
                    self.twDictProduct[pid] = [rid]
                self.twListV.append(i)

            elif v[5] >= self.twBegin + self.twSize:
                break
        print("\n****************************Begin:", self.twBegin, "Total", len(self.twListV),
              "reviews. ********************")

    def ConstructGraph_MAX(self, usingJaccard=False):
        print("Constructing reviewer graph.. usingJaccard = {}".format(usingJaccard))
        nEdges = 0
        for ri in self.reviewerGraph:
            for p in self.twDictReviewer[ri]:
                for rj in self.twDictProduct[p]:
                    if ri < rj:
                        df1 = self.gListReview[self.twDictRP[(ri, p)]][5]
                        df2 = self.gListReview[self.twDictRP[(rj, p)]][5]
                        rt1 = self.gListReview[self.twDictRP[(ri, p)]][2]
                        rt2 = self.gListReview[self.twDictRP[(rj, p)]][2]
                        if abs(df1 - df2) <= 90 and abs(rt1 - rt2) <= 3:
                            sim = self.Similarity(df1, df2, rt1, rt2)
                            if usingJaccard:
                                p1 = set(self.twDictReviewer[ri])
                                p2 = set(self.twDictReviewer[rj])
                                intersect = len(p1.intersection(p2))
                                sim *= intersect / (len(p1) + len(p2) - intersect)
                            if sim >= MIN_SIM:
                                found = False
                                for edge in self.reviewerGraph[ri][3]:
                                    if edge[0] == rj:
                                        found = True
                                        if edge[1] < sim:
                                            edge[1] = sim
                                            # update reverse edge
                                            for edgej in self.reviewerGraph[rj][3]:
                                                if edgej[0] == ri:
                                                    edgej[1] = sim
                                                    break
                                        break
                                if not found:
                                    self.Graph.add_edge(ri, rj)
                                    nEdges += 1
                                    self.reviewerGraph[ri][3].append(
                                        [rj, sim, [0, 0], [0, 0]])  # 注意：如果ri与rj共同评价多个产品，则会有多条边，但亦可进行LBP
                                    self.reviewerGraph[rj][3].append(
                                        [ri, sim, [0, 0], [0, 0]])  # neighbor, similarity, message, meesage_bak
        print("Total", nEdges, "edges in the reviewer graph.")
        fRun.writelines("Total {} edges in the reviewer graph.".format(nEdges))

    def Similarity(self, df1, df2, rt1, rt2):
        t = df1 - df2 if df1 > df2 else df2 - df1
        r = rt1 - rt2 if rt1 > rt2 else rt2 - rt1
        return 4 * PROB_TIME[t] * PROB_RATE[int(r)]

    def MRF(self):
        if USE_PRIOR:
            with open(GSMRF_HOME + FILE_PRIOR) as file:
                for index, line in enumerate(file):
                    line = line.strip().split('\t')
                    r = int(line[0])
                    prior = float(line[1])
                    if prior > 0.9999:    prior = 0.9999
                    if prior < 0.0001:    prior = 0.0001
                    self.reviewerGraph[r][0][0] = math.log(1 - prior)
                    self.reviewerGraph[r][0][1] = math.log(prior)

        groups = list(nx.connected_components(self.Graph))
        print("Total", len(self.twDictProduct), "products")
        for index, group in enumerate(groups):
            # LBP
            print("LBP... group", index, ",size", len(group))
            iter = 0
            epsilon = 1e-6
            while iter < 12:
                maxdiff = float('-Inf')
                for r in group:
                    # 计算 结点 m(k->i),k 为 i 的邻居
                    prodMsg = [0, 0]
                    for edge in self.reviewerGraph[r][3]:  # scan r's neighbors
                        edge[3][0] = edge[2][0]  # save message for latter remove
                        edge[3][1] = edge[2][1]
                        for edge2 in self.reviewerGraph[edge[0]][3]:
                            if edge2[0] == r:  # the reviewer that connects to r
                                prodMsg[0] += edge2[2][0]
                                prodMsg[1] += edge2[2][1]
                                break

                    self.reviewerGraph[r][1][0] = prodMsg[0]
                    self.reviewerGraph[r][1][1] = prodMsg[1]

                for r in group:
                    for j, edge in enumerate(self.reviewerGraph[r][3]):  # scan r's neighbors, indexed by j
                        # message(i->j)
                        for edge2 in self.reviewerGraph[edge[0]][3]:  # find the message from j to r
                            if edge2[0] == r:
                                # compute product message by removing the message from j to r
                                newMsg = [self.reviewerGraph[r][1][0] - edge2[3][0],
                                          self.reviewerGraph[r][1][1] - edge2[3][1]]
                                break

                        newMsg[0] += self.reviewerGraph[r][0][0]  # add prior potential phi(i)
                        newMsg[1] += self.reviewerGraph[r][0][1]

                        weight_ij = self.reviewerGraph[r][3][j][1]
                        term = (newMsg[0] + weight_ij,
                                newMsg[1] - weight_ij)  # 相似度加权边势 math.log(math.exp(weight_ij)) => weight_ij
                        maxterm = max(term)  # normalize
                        phi_honest = math.log(math.exp(term[0] - maxterm) + math.exp(term[1] - maxterm)) + maxterm
                        term = (newMsg[0] - weight_ij,
                                newMsg[1] + weight_ij)  # 相似度加权边势 math.log(math.exp(weight_ij)) => weight_ij
                        maxterm = max(term)  # normalize
                        phi_fraud = math.log(math.exp(term[0] - maxterm) + math.exp(term[1] - maxterm)) + maxterm

                        maxphi = max(phi_honest, phi_fraud)  # normalize (phi_honest, phi_fraud)
                        substract = math.log(math.exp(phi_honest - maxphi) + math.exp(phi_fraud - maxphi)) + maxphi
                        newMsg[0] = phi_honest - substract
                        newMsg[1] = phi_fraud - substract

                        if maxdiff < abs(edge[2][0] - newMsg[0]):
                            maxdiff = abs(edge[2][0] - newMsg[0])
                        if maxdiff < abs(edge[2][1] - newMsg[1]):
                            maxdiff = abs(edge[2][1] - newMsg[1])

                        edge[2][0] = newMsg[0]  # update message(r,rj)
                        edge[2][1] = newMsg[1]
                iter += 1
                print("iter", iter)
                if maxdiff < epsilon:
                    break

            print("\nComputing belief..")
            # belief of nodes

            for r in group:
                # 计算 结点 m(k->i),k 为 i 的邻居
                prodMsg = [0, 0]
                simSum = 0
                for edge in self.reviewerGraph[r][3]:
                    simSum += edge[1]  # weight, similarity
                    for edge2 in self.reviewerGraph[edge[0]][3]:  # the neighbor's neighbor
                        if edge2[0] == r:
                            prodMsg[0] += edge2[2][0]  # collect messages
                            prodMsg[1] += edge2[2][1]
                            break

                belief = [prodMsg[0] + self.reviewerGraph[r][0][0], prodMsg[1] + self.reviewerGraph[r][0][1]]

                maxbelief = max(belief)
                substract = math.log(math.exp(belief[0] - maxbelief) + math.exp(belief[1] - maxbelief)) + maxbelief
                self.reviewerGraph[r][2][0] = math.exp(belief[0] - substract)
                self.reviewerGraph[r][2][1] = math.exp(belief[1] - substract)
        # ends LBP

        group_size_dict = {}
        grpList = []
        for index, group in enumerate(groups):
            if len(group) not in group_size_dict:
                group_size_dict[len(group)] = []
            group_size_dict[len(group)].extend(group)
            # group details
            sumSpam = 0
            maxSpam = -math.inf
            minSpam = math.inf
            correct = 0
            for rid in group:
                self.reviewerGraph[rid][1][0] = len(group)
                self.reviewerGraph[rid][1][1] = index
                spam = self.reviewerGraph[rid][2][1]
                sumSpam += spam
                if spam > maxSpam:  maxSpam = spam
                if spam < minSpam:  minSpam = spam
                if self.reviewerGraph[rid][4] == 1: correct += 1
            grpList.append([len(group), correct / len(group), maxSpam, minSpam, sumSpam / len(group)])

        grpList.sort(key=operator.itemgetter(1, 0), reverse=True)
        grpList = pd.DataFrame(grpList)
        grpList.to_csv(ResultFile + '_GroupDetails.csv', header=None, index=None)

        for size, grp in group_size_dict.items():
            if size >= 2:
                vbelief = []
                for rid in grp:
                    pri = math.exp(self.reviewerGraph[rid][0][1])
                    spam = self.reviewerGraph[rid][2][1]
                    label = self.reviewerGraph[rid][4]
                    index = self.reviewerGraph[rid][1][1]
                    vbelief.append([rid, spam, label, len(self.reviewerGraph[rid][3]), pri, size, index])

                vbelief.sort(key=operator.itemgetter(1), reverse=True)
                self.ResultAnal(vbelief, ResultFile + "_gs" + str(size) + ".csv")

        AllBelief = []
        for r in self.reviewerGraph:
            if len(self.reviewerGraph[r][3]) > 0:  # no-neighbor nodes excluded
                pri = math.exp(self.reviewerGraph[r][0][1])
                spam = self.reviewerGraph[r][2][1]
                label = self.reviewerGraph[r][4]
                size = self.reviewerGraph[r][1][0]
                neighbor = len(self.reviewerGraph[r][3])
                index = self.reviewerGraph[r][1][1]
                AllBelief.append([r, spam, label, neighbor, pri, size, index])

        AllBelief.sort(key=operator.itemgetter(1), reverse=True)
        AllBelief = self.ResultAnal(AllBelief, ResultFile + "_allsize")

    def ResultAnal(self, vbelief, filename):
        fp = open(filename, 'w')
        fpshort = open(filename + '_short.csv', 'w')
        correct = 0
        dcg = 0
        idcg = 0
        sumAP = 0
        nAP = 0
        for i, b in enumerate(vbelief):
            idcg += 1 / math.log(i + 2, 2)
            if b[2] == 1:
                correct += 1

                dcg += 1 / math.log(i + 2, 2)
                sumAP += correct / (i + 1)
                nAP += 1
            b.extend([sumAP / nAP if nAP != 0 else 0, correct / (i + 1), dcg / idcg])

        # r, spam, label, neighbor, pri, gsize, gindex
        fp.writelines("Top\tAP\tPrecision\tNDCG\trid\t\tspam\t\tlabel\tneighbors\tprior\tgsize\tgid\n")
        fpshort.writelines("Top\tAP\tPrecision\tNDCG\trid\tspam\tlabel\tneighbors\tprior\tgsize\tgid\n")

        for k, b in enumerate(vbelief):
            # print("%4d\t%.4f\t%.4f\t%.4f" % (pos, top_APs[k][0] / top_APs[k][1], top_prec[k], top_ndcg[k]))
            fp.writelines("%4d\t%.4f\t%.4f\t%.4f\t%4d\t%.4f\t%4d\t%4d\t%.4f\t%4d\t%4d\n" % (
                k + 1, b[-3], b[-2], b[-1], b[0], b[1], b[2], b[3], b[4], b[5], b[6]))
            if (k + 1) % 100 == 0:
                fpshort.writelines("%4d\t%.4f\t%.4f\t%.4f\t%4d\t%.4f\t%4d\t%4d\t%.4f\t%4d\t%4d\n" % (
                    k + 1, b[-3], b[-2], b[-1], b[0], b[1], b[2], b[3], b[4], b[5], b[6]))
        fp.close()
        fpshort.close()


if __name__ == "__main__":  # main
    print('FILE_META:{}, FILE_PRIOR:{}'.format(FILE_META, FILE_PRIOR))
    print('DF_FROM:{}, DF_TO:{}, PRIOR_FRAUD:{}, SIGMA1:{}, SIGMA2:{}, MIN_SIM:{}'.
          format(str(DF_FROM), str(DF_TO), str(PRIOR_FRAUD), str(SIGMA1), str(SIGMA2), str(MIN_SIM)))

    gvList = []  # global review list
    DictRP = {}  # key = (rid,pid), value = a list of review index

    # read dataset
    file_name = GSMRF_HOME + FILE_META
    fp1 = open(file_name)
    listRid = []
    userLabel = dict()
    for lines in fp1.readlines():
        lines = lines.replace("\n", "")
        lines = lines.split('\t')
        v = [int(lines[0]), int(lines[1]), float(lines[2]), 1 if lines[3] == "-1" else 0,
             lines[4]]  # CUST_ID, PROD_ID, RATING, Label, fulldate, DAYDIFF

        v.append((datetime.datetime.strptime(v[4], "%Y-%m-%d") - datetime.datetime(2004, 1, 1, 0, 0, 0)).days)
        gvList.append(v)  # daydiff
        listRid.append(v[0])
        if v[0] in userLabel:
            if v[3] == 1:
                userLabel[v[0]] = 0
        else:
            userLabel[v[0]] = v[3]
    fp1.close()
    PriorFeaturesPotential = {}
    gvList.sort(key=operator.itemgetter(5))  # sort review by daydiff

    # key = (rid,pid), value = list of vids
    for i, v in enumerate(gvList):
        if (v[0], v[1]) not in DictRP:
            DictRP[(v[0], v[1])] = i
    # detect in a time window, sliding windows
    for i in range(DF_FROM, DF_TO, TW_DETECT_STEP):
        fRun = open(fRunFile, 'w')
        start = time.time()
        TWstart = time.time()
        tw = TimeWindow(i, TWSIZE, gvList, DictRP)
        tw.LoadReviews()
        tw.ConstructGraph_MAX()
        TWend = time.time()
        time1 = (TWend - TWstart) / 60
        print("Load reviews time：{} min.".format(time1))

        tw.MRF()

        end = time.time()
        time1 = (end - start) / 60
        print("Time elapsed：{} min.".format(time1))
        fRun.writelines("Time elapsed：{} min.".format(time1))
        fRun.close()
