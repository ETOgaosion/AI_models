import numpy as np
import os
import json
import sys

class PLSA:
    def __init__(self, doc_num, topic_num, word_num, n_wd, idx2word, equ_epsilon, max_itertime, hot_words_num) -> None:
        print("[PLSA] Initializing...")
        self.topic_num = topic_num  # K
        self.doc_num = doc_num  # N
        self.word_num = word_num    # M
        self.n_wd = n_wd
        self.idx2word = idx2word
        self.cur_likelihood = 0
        self.last_likelihood = 0
        self.equ_epsilon = equ_epsilon
        self.max_itertime = max_itertime
        self.hot_words_num = hot_words_num
        randmat = np.random.rand(self.topic_num,self.doc_num)
        self.p_zCd = randmat/np.sum(randmat)    # P(z|d), sum needs tp be 1, horizental is z, vertical is d
        randmat = np.random.rand(self.word_num,self.topic_num)
        self.p_wCz = randmat/np.sum(randmat)    # P(w|z), horizental is w, vertical is z
        self.q_z = np.zeros((self.topic_num,self.word_num,self.doc_num))    # Q(z) P(z|d,w), k, i, j
        self.p_wd = np.dot(self.p_wCz,self.p_zCd)   # P(d,w)
    
    def E_step(self):
        print("\t[PLSA] E step...")
        self.q_z = np.array(list(map(lambda i: np.dot(self.p_wCz[:,[i]],self.p_zCd[[i],:])/self.p_wd, np.arange(self.topic_num))))

    def M_step(self):
        print("\t[PLSA] M step...")
        p_dw_stack = np.repeat(self.n_wd[np.newaxis,:,:],self.topic_num,0)
        mid_mul = p_dw_stack * self.q_z
        self.p_wCz = (np.sum(mid_mul,2).T/np.sum(mid_mul,(1,2)))
        self.p_zCd = (np.sum(mid_mul,1)/np.sum(self.n_wd,0))
    
    def cal_joint_probability(self):
        self.p_wd = np.dot(self.p_wCz,self.p_zCd)
    
    def cal_likelihood(self):
        self.cal_joint_probability()
        log_p_wd = np.log(self.p_wd)
        if np.sum(np.isnan(log_p_wd)) + np.sum(np.isinf(log_p_wd)) == 0:
            self.cur_likelihood = np.sum(self.n_wd * log_p_wd,(0,1))
    
    def output(self):
        print("[PLSA] output result...")
        res_path = os.getcwd()+"/res"
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        doc_topic_res_path = os.path.join(res_path,"doc_topic_res.txt")
        np.savetxt(doc_topic_res_path, self.p_zCd.T)
        topic_word_res_path = os.path.join(res_path,"topic_word_res.txt")
        np.savetxt(topic_word_res_path,self.p_wCz)
        doc_word_res_path = os.path.join(res_path,"doc_word_res.txt")
        np.savetxt(doc_word_res_path,self.p_wd)
        topic_top_words = os.path.join(res_path,"topic_top_words.txt")
        if self.hot_words_num < 0 or self.hot_words_num > self.word_num:
            self.hot_words_num = self.hot_words_num//1000
        word_idx = (np.argpartition(self.p_wCz,-self.hot_words_num,0)[-self.hot_words_num:]).T
        top_words = np.array(list(map(lambda x: self.idx2word[x],word_idx.flatten().astype(str)))).reshape(np.shape(word_idx))
        np.savetxt(topic_top_words,top_words,fmt="%s")
        print("[PLSA] output completed")

    def main_func(self):
        print("[PLSA] starting...")
        end_res = " Iteration times reaches the limits."
        for iter in range(self.max_itertime):
            print(">>>[PLSA] Iteration %d" % iter)
            self.E_step()
            self.M_step()
            self.cal_likelihood()
            if np.isnan(self.cur_likelihood) or (self.last_likelihood != 0 and abs((self.cur_likelihood - self.last_likelihood)/self.last_likelihood) < self.equ_epsilon):
                end_res = " Likelihood converge."
                break
            print("current likelihood is: %f" % -self.cur_likelihood)
            print("last likelihood is: %f" % -self.last_likelihood)
            if self.last_likelihood != 0:
                print("likelihood decrease gap is: %f" % abs((self.cur_likelihood - self.last_likelihood)/self.last_likelihood))
            self.last_likelihood = self.cur_likelihood
        print("[PLSA] completed!"+end_res)
        self.output()

def read_input(data_path):
    input_path = os.getcwd()+ "/"+ data_path
    data_path = input_path + "/data.txt"
    json_path = input_path + "/voc.json"
    data = np.loadtxt(data_path)
    f = open(json_path)
    voc = json.load(f)
    idx2word = voc["idx2word"]
    return data,idx2word

def try_input(arg_str):
    print_str = "please enter the " + arg_str + " argument, default is 10:\t"
    print(print_str,end="")
    try:
        return int(input())
    except:
        print("wrong input, so we use default number")
        return 10


if __name__ == '__main__':
    data_path = sys.argv[1]
    data, idx2word = read_input(data_path)
    topic_num = try_input("topic_num")
    equ_epsilon = (10**try_input("equ_epsilon"))*(1e-15)
    max_itertime = try_input("max_itertime")
    hot_words_num = try_input("hot_words_num")
    plsa = PLSA(np.shape(data)[0],topic_num,np.shape(data)[1],data.T,idx2word,equ_epsilon,max_itertime,hot_words_num)
    plsa.main_func()
