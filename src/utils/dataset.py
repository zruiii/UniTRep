import dgl
import json
import random
import argparse
import pickle as pkl
from tqdm import tqdm
import torch
import numpy as np
from collections import defaultdict, Counter

PAD_ID = 0  # 缺失经历节点ID
END_TIME = 2022 # 默认“至今”结束时间
MIN_DUR = 3   # 3年以内的空缺合并到上一段
MAX_DUR = 8   # 8年以上的空缺经历直接切断


class HierDataset(object):
    def __init__(self, data_path, train_ratio=0.5, version=5, neg_num=500, create=True) -> None:
        self.etypes = [0, 1, 2, 3]     # e-e, w-w, e-w, w-e
        self.neg_num = neg_num
        save_path = data_path + f"hier{version}/"
        self.save_path = save_path
        
        if create:
            self.major, self.job, self.school, self.company = set(), set(), set(), set()
            self.major_map, self.job_map, self.school_map, self.company_map = dict(), dict(), dict(), dict()
            
            with open(data_path + "structure_data.pkl", "rb") as f:
                exp_dict = pkl.load(f)
                
            # split data
            train_len = int(len(exp_dict) * train_ratio)
            # val_len = int(len(exp_dict) * 0.2)
            val_len = 0     # train: test == 5:5
            user_keys = list(set(exp_dict.keys())) 
            random.shuffle(user_keys)
            train_user = user_keys[:train_len]
            val_user = user_keys[train_len:train_len + val_len]
            test_user = user_keys[train_len + val_len:]

            # construct training graph & sequence
            train_parent_g, train_child_g, train_seq, pool_map = self._create_dynamic_hete_graph(exp_dict, train_user, m=MIN_DUR, n=MAX_DUR)
            
            # sample testing data (edges & sequences)           
            parent_link_pred, child_link_pred, parent_ranking, child_ranking, traj_pred, sd_data, traj_data = self._sample_test_val(exp_dict, test_user, val_user, train_parent_g, train_child_g)
            
            with open(save_path + "train_data.pkl", "wb") as f:
                pkl.dump((train_parent_g, train_child_g, train_seq, pool_map), f, protocol = 4)
            with open(save_path + "link_pred_data.pkl", "wb") as f:
                pkl.dump((parent_link_pred, child_link_pred), f)
            with open(save_path + "ranking_data.pkl", "wb") as f:
                pkl.dump((parent_ranking, child_ranking), f)
            with open(save_path + "traj_pred_data.pkl", "wb") as f:
                pkl.dump(traj_pred, f)
            with open(save_path + "sd_data.pkl", "wb") as f:
                pkl.dump(sd_data, f)
            with open(save_path + "traj_data.pkl", "wb") as f:
                pkl.dump(traj_data, f)
                
            with open(save_path + "major_map.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.major_map))
            with open(save_path + "job_map.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.job_map))
            with open(save_path + "school_map.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.school_map))
            with open(save_path + "company_map.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.company_map))
        else:
            with open(save_path + "train_data.pkl", "rb") as f:
                train_parent_g, train_child_g, train_seq, pool_map = pkl.load(f)
            with open(save_path + "link_pred_data.pkl", "rb") as f:
                parent_link_pred, child_link_pred = pkl.load(f)
            with open(save_path + "ranking_data.pkl", "rb") as f:
                parent_ranking, child_ranking = pkl.load(f)
            with open(save_path + "traj_pred_data.pkl", "rb") as f:
                traj_pred = pkl.load(f)
            with open(save_path + "sd_data.pkl", "rb") as f:
                sd_data = pkl.load(f)
            with open(save_path + "traj_data.pkl", "rb") as f:
                traj_data = pkl.load(f)
            
            with open(save_path + "major_map.json", "r", encoding="utf-8") as f:
                self.major_map = json.load(f)
            with open(save_path + "job_map.json", "r", encoding="utf-8") as f:
                self.job_map = json.load(f)
            with open(save_path + "school_map.json", "r", encoding="utf-8") as f:
                self.school_map = json.load(f)
            with open(save_path + "company_map.json", "r", encoding="utf-8") as f:
                self.company_map = json.load(f)
        
        self.train_parent_g = train_parent_g
        self.train_child_g = train_child_g
        self.train_seq = train_seq
        self.pool_map = pool_map
        self.parent_link = parent_link_pred
        self.parent_rank = parent_ranking
        self.child_link = child_link_pred
        self.child_rank = child_ranking
        self.traj_pred = traj_pred
        self.sd_data = sd_data
        self.traj_data = traj_data


    def _create_dynamic_hete_graph(self, exp_dict, train_user, m, n):
        """ Construct Training Sequence and Graph.
        NOTE: Edge(src, dst, t, w)

        Parameters
        ----------
        exp_dict
            dict {user: {'edu_exp': , 'work_exp': }}
        train_user
            training user set
        m, optional
            将低于 n 年空缺经历视作连续, by default MIN_DUR
        n, optional
            将包含超过 m 年空缺经历的轨迹序列切开, by default MAX_DUR

        Returns
        -------
            train_parent_g: 由公司和学校构成的异构图
            train_child_g: 由 公司-岗位 和 学校-专业 构成的异构图
            train_seq: 细粒度的序列信息
            pool_map: 映射矩阵, row: School & Company;  col: Major & Job
        """

        ############ construct training graph and sequence ############
        print("Construct Training Sequence and Graph")
        print('\n')

        train_parent_s, train_child_s = self.get_sequence(train_user, exp_dict, m, n, train=True)

        school_ID = set(self.school_map.values())
        company_ID = set(self.company_map.values())
        major_ID = set(self.major_map.values())
        job_ID = set(self.job_map.values())
        
        train_parent_g = self.convert_sequence_to_graph(train_parent_s, school_ID, company_ID)
        print("# training School-Company graph")
        print(f"num nodes: {train_parent_g.num_nodes()}")
        print(f"num edges: {train_parent_g.num_edges()}")
        print(f"# school-school: {(train_parent_g.edata[dgl.ETYPE] == 0).sum().item()}")
        print(f"# company-company: {(train_parent_g.edata[dgl.ETYPE] == 1).sum().item()}")
        print(f"# school-company: {(train_parent_g.edata[dgl.ETYPE] == 2).sum().item()}")
        print(f"# company-school: {(train_parent_g.edata[dgl.ETYPE] == 3).sum().item()}")
        print(f"# unknown: {(train_parent_g.edata[dgl.ETYPE] == 4).sum().item()}")
        print('\n')
        
        train_child_g = self.convert_sequence_to_graph(train_child_s, major_ID, job_ID)
        print("# training Major-Job graph")
        print(f"num nodes: {train_child_g.num_nodes()}")
        print(f"num edges: {train_child_g.num_edges()}")
        print(f"# major-major: {(train_child_g.edata[dgl.ETYPE] == 0).sum().item()}")
        print(f"# job-job: {(train_child_g.edata[dgl.ETYPE] == 1).sum().item()}")
        print(f"# major-job: {(train_child_g.edata[dgl.ETYPE] == 2).sum().item()}")
        print(f"# job-major: {(train_child_g.edata[dgl.ETYPE] == 3).sum().item()}")
        print(f"unknown: {(train_child_g.edata[dgl.ETYPE] == 4).sum().item()}")
        print('\n')
        
        # mapping from child nodes to parent nodes
        pool_map = self._node_pool()

        return train_parent_g, train_child_g, train_child_s, pool_map


    def get_sequence(self, user_set, exp_dict, m, n, train=True):
        """ Get structured sequence of target user_set.

        Parameters
        ----------
        user_set
            _description_
        exp_dict
            _description_
        m
            _description_
        n
            _description_
        train, optional
            _description_, by default True

        Returns
        -------
            [NID, start_time, end_time]

        Raises
        ------
        ValueError
            _description_
        """
        parent_s, child_s = list(), list()
        for user in tqdm(user_set, desc="Construct {} Sequence".format("Training" if train else "Validation")):
            parent_seq, child_seq = [], []
            edu_exp, work_exp = exp_dict[user]['edu_exp'], exp_dict[user]['work_exp']
            merge_exp = sorted(edu_exp + work_exp, key=lambda x: x[0])
            
            # experience before last one
            for idx, exp in enumerate(merge_exp[:-1]):
                current_start_time = int(exp[0])
                current_end_time = int(exp[1])
                next_start_time = int(merge_exp[idx + 1][0])
                next_end_time = merge_exp[idx + 1][1]
                
                if next_end_time != "至今" and int(next_end_time) < current_end_time:
                    raise ValueError
                
                # unknown <= m : merge with the latest one
                if next_start_time - current_end_time <= m:
                    parent_src, child_src = self._node_map(exp, train)
                    parent_seq.append((parent_src, current_start_time, next_start_time))
                    child_seq.append((child_src, current_start_time, next_start_time))
                
                # unknown <= n : padding with a blank node
                elif next_start_time - current_end_time <= n:
                    parent_src, child_src = self._node_map(exp, train)
                    parent_seq.append((parent_src, current_start_time, current_end_time))
                    child_seq.append((child_src, current_start_time, current_end_time))
                    parent_seq.append((PAD_ID, current_end_time, next_start_time))
                    child_seq.append((PAD_ID, current_end_time, next_start_time))
                
                # unknown > n : split the sequence
                else:
                    if len(child_seq) == 0:           
                        continue
                    else:
                        last_parent, last_child = self._node_map(exp, train)
                        parent_seq.append((last_parent, current_start_time, current_end_time))
                        child_seq.append((last_child, current_start_time, current_end_time))
                        parent_s.append(parent_seq)
                        child_s.append(child_seq)
                        parent_seq, child_seq = [], []
            
            # last experience
            if len(child_seq) == 0:           
                continue
            else:
                last_parent, last_child = self._node_map(merge_exp[-1], train)
                if merge_exp[-1][1] != "至今":
                    parent_seq.append((last_parent, int(merge_exp[-1][0]), int(merge_exp[-1][1])))
                    child_seq.append((last_child, int(merge_exp[-1][0]), int(merge_exp[-1][1])))
                else:
                    parent_seq.append((last_parent, int(merge_exp[-1][0]), END_TIME))
                    child_seq.append((last_child, int(merge_exp[-1][0]), END_TIME))

                parent_s.append(parent_seq)
                child_s.append(child_seq)
        
        return parent_s, child_s
        

    def convert_sequence_to_graph(self, sequence, set_1, set_2):
        """ Convert the sequence data into heterogeneous graph

        Parameters
        ----------
        sequence
            sequential data
        set_1
            nodes ID (ntype=0)
        set_2
            nodes ID (ntype=1)
        self_loop
            src == dst

        Returns
        -------
            _description_
        """
        g_dict = defaultdict()
        for seq in sequence:
            for idx, exp in enumerate(seq[:-1]):
                src = exp[0]
                dst = seq[idx + 1][0]
                date = exp[2]
                if src in set_1 and dst in set_1:
                    etype = 0
                elif src in set_2 and dst in set_2:
                    etype = 1
                elif src in set_1 and dst in set_2:
                    etype = 2
                elif src in set_2 and dst in set_1:
                    etype = 3
                else:
                    etype = 4
                
                edge = tuple((src, dst, date, etype))
                if edge not in g_dict:
                    g_dict[edge] = 1
                else:
                    g_dict[edge] += 1
        
        src_li, dst_li, weight_li, date_li, etype_li = [], [], [], [], []
        for quadruple, w in g_dict.items():
            src, dst, date, etype = quadruple
            src_li.append(src)
            dst_li.append(dst)
            weight_li.append(w)
            date_li.append(date)
            etype_li.append(etype)
        
        g = dgl.graph(data=(torch.LongTensor(src_li), torch.LongTensor(dst_li)),
                      num_nodes=max(src_li + dst_li) + 1)
        g.edata['weight'] = torch.LongTensor(weight_li)
        g.edata['date'] = torch.LongTensor(date_li)
        g.edata[dgl.ETYPE] = torch.LongTensor(etype_li)
        
        ntype = []
        for node in g.nodes().tolist():
            if node in set_1:
                ntype.append(0)
            elif node in set_2:
                ntype.append(1)
            else:
                ntype.append(2)
        g.ndata['ntype'] = torch.LongTensor(ntype)
        
        return g
    
    
    def _sample_test_val(self, exp_dict, test_user, val_user, train_parent_g, train_child_g):
        """ Get Test & Valid data (edges & sequences) with consider of entities in training data only.

        Parameters
        ----------
        exp_dict
            _description_
        test_user
            _description_
        val_user
            _description_
        train_parent_g
            _description_
        train_child_g
            _description_

        Returns
        -------
            _description_
        """
        
        print("Collect Testing Data")
        test_pos_parent_pairs, test_pos_child_pairs, test_child_seq, test_parent_flow, test_child_flow = self.sample_eval(test_user, exp_dict, train_parent_g, train_child_g)
        print("\n")
        print("Collect Validation Data")
        val_pos_parent_pairs, val_pos_child_pairs, val_child_seq, val_parent_flow, val_child_flow  = self.sample_eval(val_user, exp_dict, train_parent_g, train_child_g)

        print("\n")
        print('*' * 10, 'School & Company', '*' * 10)
        print(f"# Testing school-school: {len(test_pos_parent_pairs[0])}")
        print(f"# Testing company-company: {len(test_pos_parent_pairs[1])}")
        print(f"# Testing school-company: {len(test_pos_parent_pairs[2])}")
        print(f"# Testing company-school: {len(test_pos_parent_pairs[3])}")
        print(f"# Validation school-school: {len(val_pos_parent_pairs[0])}")
        print(f"# Validation company-company: {len(val_pos_parent_pairs[1])}")
        print(f"# Validation school-company: {len(val_pos_parent_pairs[2])}")
        print(f"# Validation company-school: {len(val_pos_parent_pairs[3])}")
        print("\n")
        
        print('*' * 10, 'Major & Job', '*' * 10)
        print(f"# Testing sequence lenth: {len(test_child_seq)}")
        print(f"# Testing major-major: {len(test_pos_child_pairs[0])}")
        print(f"# Testing work-work: {len(test_pos_child_pairs[1])}")
        print(f"# Testing major-work: {len(test_pos_child_pairs[2])}")
        print(f"# Testing work-major: {len(test_pos_child_pairs[3])}")
        print(f"# Validation sequence lenth: {len(val_child_seq)}")
        print(f"# Validation major-major: {len(val_pos_child_pairs[0])}")
        print(f"# Validation job-job: {len(val_pos_child_pairs[1])}")
        print(f"# Validation major-job: {len(val_pos_child_pairs[2])}")
        print(f"# Validation job-major: {len(val_pos_child_pairs[3])}")
        print('\n')
        
        ################## Supply & Demand Raw Data ##################
        sd_data = {'in_test': dict(test_parent_flow['in'], **test_child_flow['in']),
                   'out_test': dict(test_parent_flow['out'], **test_child_flow['out']),
                   'in_val': dict(val_parent_flow['in'], **val_child_flow['in']),
                   'out_val': dict(val_parent_flow['out'], **val_child_flow['out'])}
        
        ################## Trajectory Raw Data ##################
        traj_data = {'test': test_child_seq, 'val': val_child_seq}
        
        ################## Negative Sampling for Sequence Modeling ##################
        print('\n')
        print('*' * 10, "Negative Sampling for Sequence Modeling", '*' * 10)
        traj_pred = self.get_traj_pred(train_child_g, test_child_seq, val_child_seq, k=self.neg_num)
        
        ################## Negative Sampling for Link Prediction ##################
        print('\n')
        print('*' * 10, "Negative Sampling for Link Prediction", '*' * 10)
        print('School & Company Negative Sampling')
        parent_link_pred = self.get_link_pred(train_parent_g, test_pos_parent_pairs, val_pos_parent_pairs)
        print('\n')
        print('Major & Job Negative Sampling')
        child_link_pred = self.get_link_pred(train_child_g, test_pos_child_pairs, val_pos_child_pairs)
        
        ################## Negative Sampling for Ranking ##################
        print('\n')
        print('*' * 10, "Negative Sampling for Ranking", '*' * 10)
        print('School & Company Negative Sampling')
        parent_ranking = self.get_ranking(train_parent_g, test_pos_parent_pairs, val_pos_parent_pairs, k=self.neg_num)
        print('\n')
        print('Major & Job Negative Sampling')
        child_ranking = self.get_ranking(train_child_g, test_pos_child_pairs, val_pos_child_pairs, k=self.neg_num)
        
        return parent_link_pred, child_link_pred, parent_ranking, child_ranking, traj_pred, sd_data, traj_data


    def sample_eval(self, user_set, exp_dict, train_parent_g, train_child_g, m=MIN_DUR, n=MAX_DUR, train=False):
        # check the items and filter out sequence if there is an experience out of training set
        valid_user = []
        for user in user_set:
            edu_exp, work_exp = exp_dict[user]['edu_exp'], exp_dict[user]['work_exp']
            merge_exp = sorted(edu_exp + work_exp, key=lambda x: x[0])
        
            valid = True
            for exp in merge_exp:
                parent, child = self._node_map(exp, train)
                if not parent or not child:
                    valid = False
                    break
                
            if not valid:
                continue
            else:
                valid_user.append(user)
        
        school_ID = set(self.school_map.values())
        company_ID = set(self.company_map.values())
        major_ID = set(self.major_map.values())
        job_ID = set(self.job_map.values())
        
        parent_s, child_s = self.get_sequence(valid_user, exp_dict, m, n, train)
        parent_edges, parent_flow = self.get_pos_edges(parent_s, train_parent_g, school_ID, company_ID, parent=True)
        child_edges, child_flow = self.get_pos_edges(child_s, train_child_g, major_ID, job_ID, parent=False)
        return parent_edges, child_edges, child_s, parent_flow, child_flow
    
    
    def get_pos_edges(self, sequence, g, set_1, set_2, parent=True, loop=False):
        """ extract positive edges from test/val sequence

        Parameters
        ----------
        sequence
            test or validation sequence
        g
            training graph
        set_1
            nodes ID (ntype=0)
        set_2
            nodes ID (ntype=1)
        parent, optional
            _description_, by default True
        loop, optional
            whether consider the self-loop edge (S&C Graph), by default False

        Returns
        -------
            pos_pairs: each etype with corresponding edge set.  dict({etype: [(src, dst, date), ...]})
            in_flow, out_flow: each ntype with corresponding node attribution (by year).  dict({ntype: {year: {nid: v, ...}})
        """
        pos_pairs = defaultdict(list)
        in_flow, out_flow = defaultdict(dict), defaultdict(dict)
        for seq in tqdm(sequence, desc="extract {} postive edges".format("parent" if parent else "child")):
            for idx, exp in enumerate(seq[:-1]):
                src = exp[0]
                dst = seq[idx + 1][0]
                date = exp[2]
                if not loop and src == dst:
                    continue
                
                # nodes in training graph are considered
                if not parent:
                    name_1, name_2 = 'major', 'job'
                else:
                    name_1, name_2 = 'school', 'company'
                
                if src in set_1:
                    if name_1 not in out_flow:
                        out_flow[name_1] = defaultdict(dict)

                    if src not in out_flow[name_1][date]:
                        out_flow[name_1][date][src] = 1
                    else:
                        out_flow[name_1][date][src] += 1
                        
                if src in set_2:
                    if name_2 not in out_flow:
                        out_flow[name_2] = defaultdict(dict)
                        
                    if src not in out_flow[name_2][date]:
                        out_flow[name_2][date][src] = 1
                    else:
                        out_flow[name_2][date][src] += 1
                
                if dst in set_1:
                    if name_1 not in in_flow:
                        in_flow[name_1] = defaultdict(dict)
                        
                    if dst not in in_flow[name_1][date]:
                        in_flow[name_1][date][dst] = 1
                    else:
                        in_flow[name_1][date][dst] += 1
                
                if dst in set_2:
                    if name_2 not in in_flow:
                        in_flow[name_2] = defaultdict(dict)
                        
                    if dst not in in_flow[name_2][date]:
                        in_flow[name_2][date][dst] = 1
                    else:
                        in_flow[name_2][date][dst] += 1
                
                # edge in training graph is positive
                if src in set_1 and dst in set_1:
                    etype = 0
                elif src in set_2 and dst in set_2:
                    etype = 1
                elif src in set_1 and dst in set_2:
                    etype = 2
                elif src in set_2 and dst in set_1:
                    etype = 3
                else:
                    continue
                
                if not g.has_edges_between(src, dst):
                    pos_pairs[etype].append((src, dst, date))
        
        for etype, tup in pos_pairs.items():
            pos_pairs[etype] = list(set(tup))
        return pos_pairs, {'in': in_flow, 'out': out_flow}
        
        
    def get_link_pred(self, train_g, test_pos_edges, val_pos_edges):
        """ 1:1 Sampling for Link Prediction

        Parameters
        ----------
        train_g
            _description_
        test_pos
            _description_
        val_pos
            _description_

        Returns
        -------
        dict({etype: link predition data})
        """
        res_data = defaultdict(dict)
        train_src, train_dst = train_g.edges()
        ntype = train_g.ndata['ntype'].numpy()
        train_nodes = train_g.nodes().numpy()
        node_dict = {nt: train_nodes[ntype == nt] for nt in list(set(ntype))}   # ntype: node_set
        
        for etype in self.etypes:
            # all positive edges & candidate nodes
            test_pos = test_pos_edges[etype]
            val_pos = val_pos_edges[etype]
            index = torch.where(train_g.edata[dgl.ETYPE] == etype)
            all_pos_pairs = set(list(zip(train_src[index].numpy(), train_dst[index].numpy())) + \
                                [(x[0], x[1]) for x in test_pos] + \
                                [(x[0], x[1]) for x in val_pos])
            
            pos_pairs_src, pos_pairs_dst = defaultdict(set), defaultdict(set)
            for key, value in list(all_pos_pairs):
                pos_pairs_src[key].add(value)
                pos_pairs_dst[value].add(key)
            
            # random choose src/dst to obtain negative edges
            test_neg = []
            test_choices = np.random.uniform(size=len(test_pos))
            for idx, edge in enumerate(tqdm(test_pos, desc=f"sample negative testing edges-{etype}")):
                src, dst, date = edge
                if test_choices[idx] > 0.5:
                    neg_src = random.choice(node_dict[ntype[src]])
                    while neg_src in pos_pairs_dst[dst]:
                        neg_src = random.choice(node_dict[ntype[src]])
                    test_neg.append((neg_src, dst, date))
                    
                else:
                    neg_dst = random.choice(node_dict[ntype[dst]])
                    while neg_dst in pos_pairs_src[src]:
                        neg_dst = random.choice(node_dict[ntype[dst]])
                    test_neg.append((src, neg_dst, date))
            
            val_neg = []
            val_choices = np.random.uniform(size=len(val_pos))
            for idx, edge in enumerate(tqdm(val_pos, desc=f"sample negative validation edges-{etype}")):
                src, dst, date = edge
                if val_choices[idx] > 0.5:
                    neg_src = random.choice(node_dict[ntype[src]])
                    while neg_src in pos_pairs_dst[dst]:
                        neg_src = random.choice(node_dict[ntype[src]])
                    val_neg.append((neg_src, dst, date))
                else:
                    neg_dst = random.choice(node_dict[ntype[dst]])
                    while neg_dst in pos_pairs_src[src]:
                        neg_dst = random.choice(node_dict[ntype[dst]])
                    val_neg.append((src, neg_dst, date))

            res_data[etype] = {
                "test_pos_edges": np.array(test_pos),            # [E, 3]
                "test_neg_edges": np.array(test_neg),            # [E, 3]
                "val_pos_edges": np.array(val_pos),              # [E, 3]
                "val_neg_edges": np.array(val_neg)               # [E, 3]
            }
            
        return res_data
    
    
    def get_ranking(self, train_g, test_pos_edges, val_pos_edges, k=10):
        """ 1:k Destination Sampling for Ranking evaluation on target nodes (outflow)

        Parameters
        ----------
        train_g
            _description_
        test_pos_edges
            _description_
        val_pos_edges
            _description_
        k, optional
            _description_, by default 1000
            
        Returns
        ----------
        dict({etype: ranking data})  
        """
        res_data = defaultdict(dict)
        train_src, train_dst = train_g.edges()
        etype2dst = {0: 0, 1: 1, 2: 1, 3: 0}        # destination types of different etypes
        
        for etype in self.etypes:
            # all positive edges & candidate nodes 
            index = torch.where(train_g.edata[dgl.ETYPE] == etype)
            candi_nodes = train_g.nodes()[train_g.ndata['ntype'] == etype2dst[etype]].tolist()
            
            pos_pairs_src = defaultdict(set)
            all_pos_pairs = set(list(zip(train_src[index].numpy(), train_dst[index].numpy())) + \
                                [(x[0], x[1]) for x in test_pos_edges[etype]] + \
                                [(x[0], x[1]) for x in val_pos_edges[etype]])
            for key, value in tqdm(list(all_pos_pairs)):
                pos_pairs_src[key].add(value)
            
            # testing data
            test_src, test_dst, test_neg_dst = [], [], []
            for _, edge in enumerate(tqdm(test_pos_edges[etype], desc=f"sample {k} negative testing edges-{etype}")):
                src, dst, date = edge
                test_src.append((src, date))
                test_dst.append(dst)
                
                neg_dst = random.sample(candi_nodes, k * 2)
                neg_dst = set(neg_dst) - set(pos_pairs_src[src])
                test_neg_dst.append(list(neg_dst)[:k])
                
            # validation data
            val_src, val_dst, val_neg_dst = [], [], []
            for _, edge in enumerate(tqdm(val_pos_edges[etype], desc=f"sample {k} negative validation edges-{etype}")):
                src, dst, date = edge
                val_src.append((src, date))
                val_dst.append(dst)
                
                neg_dst = random.sample(candi_nodes, k * 2)
                neg_dst = set(neg_dst) - set(pos_pairs_src[src])
                val_neg_dst.append(list(neg_dst)[:k])
            
            res_data[etype] = {
                "test_src": np.array(test_src),             # [E, 2]
                "test_dst": np.array(test_dst),             # (E, )
                "test_neg_dst": np.array(test_neg_dst),     # (E, k)
                "val_src": np.array(val_src),               
                "val_dst": np.array(val_dst),
                "val_neg_dst": np.array(val_neg_dst) 
            }
        return res_data


    def get_traj_pred(self, train_g, test_seq, val_seq, k):
        ntype = train_g.ndata['ntype'].numpy()
        train_nodes = train_g.nodes().numpy()
        node_dict = {nt: train_nodes[ntype == nt].tolist() for nt in list(set(ntype))}
        
        # testing data
        test_his, test_dst, test_neg_dst = [], [], []
        for seq in tqdm(test_seq, desc="Sampling for Trajectory Prediction - Testing"):
            if len([x[0] for x in seq if x[0] != 0]) < 3:
                continue
            elif seq[-1][0] != 0:
                test_his.append(seq[:-1])
                test_dst.append(seq[-1][0])
                
                node_type = ntype[seq[-1][0]]
                neg_dst = set(random.sample(node_dict[node_type], k + 1)) - set({seq[-1][0]})
                test_neg_dst.append(list(neg_dst)[:k])
            else:
                raise ValueError
        
        # validation data
        val_his, val_dst, val_neg_dst = [], [], []
        for seq in tqdm(val_seq, desc="Sampling for Trajectory Prediction - Validation"):
            if len([x[0] for x in seq if x[0] != 0]) < 3:
                continue
            elif seq[-1][0] != 0:
                val_his.append(seq[:-1])
                val_dst.append(seq[-1][0])
                
                node_type = ntype[seq[-1][0]]
                neg_dst = set(random.sample(node_dict[node_type], k + 1)) - set({seq[-1][0]})
                val_neg_dst.append(list(neg_dst)[:k])
            else:
                raise ValueError
        
        print(f"# Testing Trajectories: {len(test_his)}")
        print(f"# Validation Trajectories: {len(val_his)}")
        res_data = {
            "test_seq": test_his,
            "test_dst": test_dst,
            "test_neg_dst": test_neg_dst,
            "val_seq": val_his,
            "val_dst": val_dst,
            "val_neg_dst": val_neg_dst,
        }
        
        return res_data
    
    
    def _node_map(self, exp, train=True):
        """ Node ID Mapping. If the entities in exp are not in train data, return None.
        NOTE: There may be school and company with the same name
        
        Parameters
        ----------
        exp
            _description_
        train
            If True, update the node set.

        Returns
        -------
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if exp[-1] == 0:
            _, _, school, major, _, _ = exp
            
            # school
            if school not in self.school:
                if train:
                    res_parent_node = len(self.school_map) + len(self.company_map) + 1
                    self.school_map[school] = res_parent_node
                    self.school.add(school)
                else:
                    return None, None
            else:
                res_parent_node = self.school_map[school]
                
            # (school, major)
            major_node = ",".join([school, major])
            if major_node not in self.major:
                if train:
                    res_child_node = len(self.major_map) + len(self.job_map) + 1
                    self.major_map[major_node] = res_child_node
                    self.major.add(major_node)
                else:
                    return None, None
            else:
                res_child_node = self.major_map[major_node]

        elif exp[-1] == 1:
            _, _, company, job, _, _ = exp

            # company
            if company not in self.company:
                if train:
                    res_parent_node = len(self.school_map) + len(self.company_map) + 1
                    self.company_map[company] = res_parent_node
                    self.company.add(company)
                else:
                    return None, None
            else:
                res_parent_node = self.company_map[company]
            
            # (company, job)
            job_node = ",".join([company, job])
            if job_node not in self.job:
                if train:
                    res_child_node = len(self.major_map) + len(self.job_map) + 1
                    self.job_map[job_node] = res_child_node
                    self.job.add(job_node)
                else:
                    return None, None
            else:
                res_child_node = self.job_map[job_node]
        else:
            raise ValueError

        return res_parent_node, res_child_node
    

    def _node_pool(self):
        """ Construct the mapping matrix: Row (Company & School) ; Column (Job & Major)

        Parameters
        ----------
        parent_map
            _description_
        child_map
            _description_

        Returns
        -------
            _description_
        """
        indices = [(0, 0)]
        
        for major, major_id in self.major_map.items():
            school, _ = major.split(',', 1)
            school_id = self.school_map[school]
            indices.append((school_id, major_id))
            
        for job, job_id in self.job_map.items():
            company, _ = job.split(',', 1)
            company_id = self.company_map[company]
            indices.append((company_id, job_id))
        
        indices = torch.LongTensor(list(set(indices))).T
        values = torch.ones(indices.shape[1])
        pool_map = torch.sparse_coo_tensor(indices, values)
        
        assert pool_map.size() == (len(self.company) + len(self.school) + 1, len(self.job) + len(self.major) + 1)
        return pool_map


