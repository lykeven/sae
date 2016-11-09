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

typedef vector<vector<double>> Vec;
typedef vector<vector<int>> Lab;
typedef pair<double, double> Res;

Test_Embedding::Test_Embedding(MappedGraph *graph)
    : Solver(graph)
{
}

Test_Embedding::~Test_Embedding()
{
}

vector<double> logistic_regression(Vec &data, vector<int> &label)
{
    double alpha = 0.025;
    int maxCycles = 200;
    double lambda = 0.1;
    int n = data[0].size(), m = data.size();
    vector<double> weights(n), delta(n);
    for (int k = 0; k < maxCycles; k++)
    {
        vector<double> h(m, 0);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
                h[i] += data[i][j] * weights[j];
            h[i] = 1.0 / (1 + exp(-h[i]));
            h[i] = label[i] - h[i];
        }
        for (int i = 0; i < n; i++)
        {
            delta[i] = 0;
            for (int j = 0; j < m; j++)
                delta[i] += h[j] * data[j][i] - lambda * weights[i] / m * (i == 0);
            weights[i] += delta[i] * alpha;
        }
    }
    //cout<<"weight"<<endl;for(int i =1; i< m ;i++) cout<<weights[i] << " "<<endl;
    return weights;
}

vector<vector<double>> multi_label_classification(Vec &data, Lab &label)
{
    int m = data.size(), n = data[0].size(), K = label.size();
    Vec theta(K, vector<double>(n, 1.0));
    for (int i = 0; i < K; i++)
        theta[i] = logistic_regression(data, label[i]);
    return theta;
}

vector<int> predict(const Vec &theta, const vector<double> &property, int pred_num)
{
    int K = theta.size(), n = property.size();
    vector<pair<double, int>> pred;
    vector<int> index(K, 0);
    for (int i = 0; i < K; i++)
    {
        double pre = 0;
        for (int j = 0; j < n; j++)
            pre += property[j] * theta[i][j];
        pre = 1.0 / (1 + exp(-pre));
        pred.push_back(make_pair(pre, i));
    }
    sort(pred.begin(), pred.end());
    reverse(pred.begin(), pred.end());
    for (int i = 0; i < pred_num; i++)
        index[pred[i].second] = 1;
    return index;
}

pair<double, double> compute_accuracy(pair<pair<Vec, Vec>, pair<Lab, Lab>> &all_data)
{
    Vec data_train = all_data.first.first, data_test = all_data.first.second;
    Lab label_train = all_data.second.first, label_test = all_data.second.second;
    Vec weight = multi_label_classification(data_train, label_train);
    int K = label_train.size(), m = data_test.size();
    vector<double> tp(K, 0), fn(K, 0), fp(K, 0), tn(K, 0);
    for (int i = 0; i < m; i++)
    {
        int pred_num = 0;
        for (int j = 0; j < K; j++)
            if (label_test[j][i] == 1)
                pred_num++;
        vector<int> pre_class = predict(weight, data_test[i], pred_num);
        for (int j = 0; j < K; j++)
        {
            //printf("real label:%d \t pred label:%d\n",label[j][i],pre_class[j]);
            if (pre_class[j] == label_test[j][i] && pre_class[j] == 1)
                tp[j]++;
            if (pre_class[j] != label_test[j][i] && pre_class[j] == 0)
                fn[j]++;
            if (pre_class[j] != label_test[j][i] && pre_class[j] == 1)
                fp[j]++;
            else
                tn[j]++;
        }
    }
    double macro_f1 = 0, precision = 0, recall = 0;
    for (int j = 0; j < K; j++)
    {
        precision = tp[j] / (tp[j] + fp[j]);
        recall = tp[j] / (tp[j] + fn[j]);
        //printf("label:%d\ta:%lf\tb:%lf\tc:%lf\td:%.lf\tpre:%lf\trec:%lf\n",j,a[j],b[j],c[j],d[j],precision,recall);
        if (precision != 0 && recall != 0 && precision == precision && recall == recall)
            macro_f1 += 2 * precision * recall / (precision + recall);
    }
    macro_f1 /= K;
    double tp_temp = accumulate(tp.begin(), tp.end(), 0);
    double fn_temp = accumulate(fn.begin(), fn.end(), 0);
    double fp_temp = accumulate(fp.begin(), fp.end(), 0);
    double micro_f1 = 2 * tp_temp * tp_temp / (2 * tp_temp * tp_temp + tp_temp * fn_temp + tp_temp * fp_temp);
    //printf("macro_f1:%.3f\tmicro_f1:%.3f\n",macro_f1,micro_f1);
    return make_pair(micro_f1, macro_f1);
}

Lab ReadLabel(string label_file, map<vid_t, vid_t> &mapToSae)
{
    int m = mapToSae.size();
    ifstream fin((label_file).c_str());
    map<int, int> label_map;
    int x, y, K = 0;
    while (1)
    {
        if (!(fin >> x >> y))
            break;
        if (label_map.find(y) == label_map.end())
            label_map.insert(make_pair(y, label_map.size()));
    }
    K = label_map.size();
    vector<int> temp(m, 0);
    vector<vector<int>> label(K, temp);
    ifstream fin2((label_file).c_str());
    while (1)
    {
        if (!(fin2 >> x >> y))
            break;
        //printf("real label:%d\t map label:%d\n",y, label_map[y]);
        label[label_map[y]][mapToSae[x]] = 1;
    }
    return label;
}

void write_data(Vec &data, Lab &label, string emb_file, string lab_file)
{
    int m = data.size(), d = data[0].size(), K = label.size();
    FILE *emb = fopen(emb_file.c_str(), "wb");
    FILE *lab = fopen(lab_file.c_str(), "wb");
    fprintf(emb, "%d %d %d\n", m, d, K);
    for (int i = 0; i < m; i++)
    {
        fprintf(emb, "%d", i);
        for (int j = 0; j < d; j++)
            fprintf(emb, " %lf", data[i][j]);
        fprintf(emb, "\n");
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < K; j++)
            fprintf(lab, "%d ", label[j][i]);
        fprintf(lab, "\n");
    }
    fclose(emb);
    fclose(lab);
    printf("embedding and file have been written into file\n");
}

pair<pair<Vec, Vec>, pair<Lab, Lab>> split_data(Vec &data, Lab &label, double ratio)
{
    int m = data.size(), d = data[0].size(), K = label.size();
    vector<int> rand_seq(m, 0);
    for (int i = 0; i < m; i++)
        rand_seq[i] = i;
    int Num = m * ratio;
    for (int j = m - 1; j >= 0; j--)
        swap(rand_seq[rand() % (m - 1)], rand_seq[j]);

    Vec data_train, data_test;
    Lab label_train, label_test;
    for (int i = 0; i < Num; i++)
        data_train.push_back(data[rand_seq[i]]);
    for (int i = Num; i < m; i++)
        data_test.push_back(data[rand_seq[i]]);
    for (int j = 0; j < K; j++)
    {
        vector<int> temp_train;
        for (int i = 0; i < Num; i++)
            temp_train.push_back(label[j][rand_seq[i]]);
        label_train.push_back(temp_train);
    }
    for (int j = 0; j < K; j++)
    {
        vector<int> temp_test;
        for (int i = Num; i < m; i++)
            temp_test.push_back(label[j][rand_seq[i]]);
        label_test.push_back(temp_test);
    }
    return make_pair(make_pair(data_train, data_test), make_pair(label_train, label_test));
}

vector<Res> test_algorithm(Vec &data, Lab &label, vector<double> ratios, int shuffle_num)
{
    vector<Res> accuracies;
    for (int i = 0; i < ratios.size(); i++)
    {
        Res accuracy = make_pair(0, 0);
        for (int j = 0; j < shuffle_num; j++)
        {
            pair<pair<Vec, Vec>, pair<Lab, Lab>> all_data = split_data(data, label, ratios[i]);
            Res accuracy_temp = compute_accuracy(all_data);
            accuracy.first += accuracy_temp.first;
            accuracy.second += accuracy_temp.second;
        }
        accuracy.first /= shuffle_num;
        accuracy.second /= shuffle_num;
        printf("ratio:%.3f\tmicro-f1:%.3f\tmacro_f1:%.3f\n", ratios[i], accuracy.first, accuracy.second);
        accuracies.push_back(accuracy);
    }
    return accuracies;
}

int Test_Embedding::solve(int R, int T, int d, int w, int S, int Th, int Neg, double init_rate, map<vid_t, vid_t> mapToSae, string label_file, string output_file)
{
    srand(time(NULL));
    vector<double> ratios = vector<double>{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    int shuffle_num = 3;
    string emb_file = "./output/embeddings";
    string lab_file = "./output/label";
    Lab label = ReadLabel(label_file, mapToSae);
    Vec data;
    vector<vector<Res>> all_accuracies(3);

    Deep_Walk dw(graph);
    data = dw.solve(R, T, d, w, Th, Neg, init_rate);
    write_data(data, label, emb_file + "1", lab_file + "1");
    all_accuracies[0] = test_algorithm(data, label, ratios, shuffle_num);

    Node2Vec nv(graph);
    data = nv.solve(R, T, d, w, Th, Neg, init_rate);
    write_data(data, label, emb_file + "2", lab_file + "2");
    all_accuracies[1] = test_algorithm(data, label, ratios, shuffle_num);

    LINE ln(graph);
    data = ln.solve(S, d / 2, Th, Neg, init_rate);
    write_data(data, label, emb_file + "3", lab_file + "3");
    all_accuracies[2] = test_algorithm(data, label, ratios, shuffle_num);

    FILE *fo = fopen(output_file.c_str(), "wb");
    fprintf(fo, "Algorithm performace\nAlgorithm 1:Deepwalk\tAlgorithm 2:Node2Vec\tAlgorithm 3:LINE\n");
    for (int i = 0; i < all_accuracies.size(); i++)
    {
        fprintf(fo, "Algorithm %d\npercent\tMicro-F1\tMacro-F1\n", i+1);
        for (int j = 0; j < all_accuracies[i].size(); j++)
            fprintf(fo, "%.3f\t%.3f\t%.3f\n", ratios[j], all_accuracies[i][j].first, all_accuracies[i][j].second);
    }
    fclose(fo);
    return 0;
}
