#include "deepwalk.h"
#include "node2vec.h"
#include "line.h"
#include "test_embedding.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cassert>
#include <iostream>
#include <fstream>
#include <map>

using namespace std;
using namespace sae::io;

Test_Embedding::Test_Embedding(MappedGraph *graph)
	:Solver(graph) {
}

Test_Embedding::~Test_Embedding() {
}


vector<double> logistic_regression(vector<vector<double> >& data, vector<int> label)
{
    double alpha = 0.1;
    int maxCycles = 1000;
    int n = data[0].size(), m = data.size();
    vector<double> weights(n), delta(n);
    for(int k = 0; k < maxCycles; k++){
        vector<double> h(m, 0);
        for(int i = 0;i<m;i++){
            for(int j = 0; j<n; j++)
                h[i] += data[i][j]* weights[j];
            h[i] = 1.0 / (1 + exp(-h[i]));
            h[i] = label[i] - h[i];
        }
        for(int i = 1;i<n;i++){
            delta[i] = 0;
            for(int j = 0;j < m;j++)
                delta[i] += h[j]* data[j][i];
            weights[i] += delta[i] * alpha;
        }
    }
    //for(int i =1; i< m ;i++) cout<<weights[i] << " "<<endl;
    return weights;
}


vector<vector<double> > multi_label_regression(vector< vector<double> >& data, vector<vector<int>> label)
{
    double alpha = 0.1;
    int maxCycles = 1000;
    int m = data.size(), n = data[0].size(), K = label.size();;
    vector<double> temp2 (n,1.0);
    vector<vector<double>> theta(K,temp2);
    for(int i=0;i<K;i++)
        theta[i] = logistic_regression(data, label[i]);
    return theta;
}

vector<int> predict(const vector<vector<double>> & theta, const vector<double> & property)
{
    int K = theta.size(), n = property.size();
    vector<double> pred (K,0);
    vector<int>index(K,0);
    for(int i=0;i<K;i++)
    {
        for(int j = 0;j<n;j++)
            pred[i] += property[j] * theta[i][j];
        pred[i] = 1.0/(1 + exp(-pred[i]));
        if(pred[i]>0.5)
            index[i] = 1;
    }
    return index;
}

pair<double,double> compute_accuracy(vector<vector<double>> data,vector<vector<int>> label)
{
    vector<vector<double> > weight = multi_label_regression(data,label);
    int K = label.size(),m = data.size();
    vector<double> a(K,0),b(K,0),c(K,0);
    for(int i=0;i<m;i++)
    {
        vector<int> pre_class = predict(weight,data[i]);
        for(int j=0;j<K;j++)
        {
            //printf("real label:%d \t pred label:%d\n",label[j][i],pre_class[j]);
            if (pre_class[j] == label[j][i]&&pre_class[j]==1)
                a[j] ++;
            if (pre_class[j] != label[j][i]&&pre_class[j]==1)
                b[j] ++;
            if (pre_class[j] != label[j][i]&&pre_class[j]==0)
                c[j] ++;
        }
    }
    double macro_f1=0, precision=0, recall=0;
    for(int j=0;j<K;j++)
    {
        precision = a[j]/(a[j]+b[j]);
        recall = a[j]/(a[j]+c[j]);
        printf("precison:%.3f\trecall:%.3f\n",precision,recall);
        macro_f1 += 2*precision*recall/(precision+recall);
    }
    macro_f1/=K;
    double a_temp = accumulate(a.begin(),a.end(),0);
    double b_temp = accumulate(b.begin(),b.end(),0);
    double c_temp = accumulate(c.begin(),c.end(),0);
    double micro_f1 = 2*a_temp*a_temp/(2*a_temp*a_temp + a_temp*b_temp + a_temp*c_temp);
    printf("macro_f1:%.3f\tmicro_f1:%.3f\n",macro_f1,micro_f1);    
    return make_pair(macro_f1,micro_f1);
}

vector<vector<int>> ReadLabel(string label_file,int m,map<vid_t, vid_t> mapToSae)
{
    ifstream fin((label_file).c_str());
    int x,y,K=0;
    while(1){
        if (!(fin >> x >> y)) break;
        K = max(K, y);
    }
    vector<int> temp (m,0);
    vector<vector<int>> label(K,temp);
    ifstream fin2((label_file).c_str());
    while(1){
        if (!(fin2 >> x >> y)) break;
        label[y-1][mapToSae[x]] = 1;
    }
    return label;
}

vector<pair<double,double>>  Test_Embedding::solve(int R, int T, int d, int w, int S, map<vid_t, vid_t> mapToSae,string label_file,string output_file)
{
    srand(time(NULL));
    time_t start_time1 = clock();
    Deep_Walk dw(graph);
    vector<vector<double> >  ans1 = dw.solve(R,T,d,w);
    time_t end_time1 = clock();

    time_t start_time2 = clock();
    Node2Vec nv(graph);
    vector<vector<double> >  ans2 = nv.solve(R,T,d,w);
    time_t end_time2 = clock();
  
    time_t start_time3 = clock();
    LINE ln(graph);
    vector<vector<double> >  ans3 = ln.solve(S,d/2);
    time_t end_time3 = clock();
    
    int m = ans1.size();
    vector<vector<int>>label = ReadLabel(label_file,m,mapToSae);
    vector<pair<double,double>> accuracy(3);
    accuracy[0] = compute_accuracy(ans1,label);
    accuracy[1] = compute_accuracy(ans2,label);
    accuracy[2] = compute_accuracy(ans3,label);
    
    
    FILE *fo = fopen(output_file.c_str(), "wb");
    fprintf(fo, "Algorithm\tMacro-F1\tMicro-F1\tRuning Time\n");
    fprintf(fo, "DeepWalk\t%.3f\t%.3f\t%.3f\n", accuracy[0].first,accuracy[0].second,(end_time1 - start_time1 + 0.0) / CLOCKS_PER_SEC);
    fprintf(fo, "Node2Vec\t%.3f\t%.3f\t%.3f\n", accuracy[1].first,accuracy[1].second,(end_time2 - start_time2 + 0.0) / CLOCKS_PER_SEC);
    fprintf(fo, "LINE\t%.3f\t%.3f\t%.3f\n", accuracy[2].first,accuracy[2].second,(end_time3 - start_time3 + 0.0) / CLOCKS_PER_SEC);
    fclose(fo);
    return accuracy;
}
