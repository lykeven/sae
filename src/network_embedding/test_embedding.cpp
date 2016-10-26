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
    double alpha = 0.025;
    int maxCycles = 100;
    double lambda = 0.1;
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
        for(int i = 0;i<n;i++){
            delta[i] = 0;
            for(int j = 0;j < m;j++)
                delta[i] += h[j]* data[j][i] - lambda*weights[i]/m*(i==0);
            weights[i] += delta[i] * alpha;
        }
    }
	//cout<<"weight"<<endl;for(int i =1; i< m ;i++) cout<<weights[i] << " "<<endl;
    return weights;
}


vector<vector<double> > multi_label_regression(vector< vector<double> >& data, vector<vector<int>> label)
{
    int m = data.size(), n = data[0].size(), K = label.size();;
    vector<double> temp2 (n,1.0);
    vector<vector<double>> theta(K,temp2);
    for(int i=0;i<K;i++)
        theta[i] = logistic_regression(data, label[i]);
    return theta;
}

vector<int> predict(const vector<vector<double>> & theta, const vector<double> & property,int pred_num)
{
    int K = theta.size(), n = property.size();
    vector<pair<double,int>> pred;
    vector<int>index(K,0);
    for(int i=0;i<K;i++)
    {
        double pre = 0;
        for(int j = 0;j<n;j++)
            pre += property[j] * theta[i][j];
        pre = 1.0/(1 + exp(-pre));
        pred.push_back(make_pair(pre, i));
    }
    sort(pred.begin(),pred.end());
    reverse(pred.begin(),pred.end());
    for(int i=0;i<pred_num;i++)
        index[pred[i].second] = 1;
    return index;
}

pair<double,double> compute_accuracy(vector<vector<double>> data,vector<vector<int>> label_train,vector<vector<int>> label_test)
{
    vector<vector<double> > weight = multi_label_regression(data,label_train);
    int K = label_train.size(),m = data.size();
    vector<double> a(K,0),b(K,0),c(K,0),d(K,0);
    for(int i=0;i<m;i++)
    {
        int pred_num = 0;
        for(int j=0;j<K;j++)
            if(label_test[j][i]==1)
                pred_num++;
        vector<int> pre_class = predict(weight,data[i],pred_num);
        for(int j=0;j<K;j++)
        {
            //printf("real label:%d \t pred label:%d\n",label[j][i],pre_class[j]);
            if (pre_class[j] == label_test[j][i]&&pre_class[j]==1)
                a[j] ++;
            if (pre_class[j] != label_test[j][i]&&pre_class[j]==0)
                b[j] ++;
            if (pre_class[j] != label_test[j][i]&&pre_class[j]==1)
                c[j] ++;
			else
				d[j] ++;
        }
    }
    double macro_f1=0, precision=0, recall=0;
    for(int j=0;j<K;j++)
    {
        precision = a[j]/(a[j]+b[j]);
        recall = a[j]/(a[j]+c[j]);
        printf("label:%d\ta:%lf\tb:%lf\tc:%lf\td:%.lf\tpre:%lf\trec:%lf\n",j,a[j],b[j],c[j],d[j],precision,recall);
        if(precision!=0&&recall!=0&&precision==precision&&recall==recall)
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

pair<vector<vector<int>>,vector<vector<int>>> ReadLabel(string label_file,int m,map<vid_t, vid_t> mapToSae,double ratio)
{
    ifstream fin((label_file).c_str());
    int x,y,K=0;
    while(1){
        if (!(fin >> x >> y)) break;
        K = max(K, y);
    }
    vector<int> temp (m,0);
    vector<vector<int>> label_train(K,temp),label_test(K,temp);
    ifstream fin2((label_file).c_str());
    while(1){
        if (!(fin2 >> x >> y)) break;
        rand()%100>ratio?label_train[y-1][mapToSae[x]] = 1:label_test[y-1][mapToSae[x]] = 1;
    }
    return make_pair(label_train,label_test);
}

vector<pair<double,double>>  Test_Embedding::solve(int R, int T, int d, int w, int S, map<vid_t, vid_t> mapToSae,string label_file,string output_file)
{
    srand(time(NULL));

    time_t start_time1 = clock();
    Deep_Walk dw(graph);
    vector<vector<double> >  ans1 = dw.solve(R,T,d,w);
    time_t end_time1 = clock();
	/*
    time_t start_time2 = clock();
    Node2Vec nv(graph);
    vector<vector<double> >  ans2 = nv.solve(R,T,d,w);
    time_t end_time2 = clock();

    time_t start_time3 = clock();
    LINE ln(graph);
    vector<vector<double> >  ans3 = ln.solve(S,d/2);
    time_t end_time3 = clock();
    */
    int m = graph->VertexCount();
    double ratio = 50;
    string train_file = "./output/train_file";
    string test_file = "./output/test_file";
    pair<vector<vector<int>>,vector<vector<int>>>label = ReadLabel(label_file,m,mapToSae,ratio);
    vector<vector<int>>label_train = label.first, label_test = label.second;
    vector<pair<double,double>> accuracy(3);
    accuracy[0] = compute_accuracy(ans1, label_train, label_test);
    //accuracy[1] = compute_accuracy(ans2, label_train, label_test);
    //accuracy[2] = compute_accuracy(ans3, label_train, label_test);

    FILE *fo = fopen(output_file.c_str(), "wb");
    fprintf(fo, "Algorithm\tMacro-F1\tMicro-F1\tRuning Time\n");
    fprintf(fo, "DeepWalk\t%.3f\t%.3f\t%.3f\n", accuracy[0].first,accuracy[0].second,(end_time1 - start_time1 + 0.0) / CLOCKS_PER_SEC);
    //fprintf(fo, "Node2Vec\t%.3f\t%.3f\t%.3f\n", accuracy[1].first,accuracy[1].second,(end_time2 - start_time2 + 0.0) / CLOCKS_PER_SEC);
    //fprintf(fo, "LINE\t%.3f\t%.3f\t%.3f\n", accuracy[2].first,accuracy[2].second,(end_time3 - start_time3 + 0.0) / CLOCKS_PER_SEC);
    fclose(fo);
    return accuracy;
}
