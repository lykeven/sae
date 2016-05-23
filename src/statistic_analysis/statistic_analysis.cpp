#include "statistic_analysis.h"
#include <iostream>
using namespace std;
namespace statistic{
    vector<double> logistic_regression
        (const std::vector< std::vector<double> >& data, const std::vector<bool> is_treated)
    {
  /*      for(int i = 0;i< data.size();i++){
            for(int j = 0;j< data[i].size();j++)
                cerr<< data[i][j] <<" ";
            cerr<<endl;
        }*/
        double alpha = 0.1;
        int maxCycles = 1000;
        int m = data[0].size(), n = data.size();
        vector<double> weights(m), delta(m);
        for(int k = 0; k < maxCycles; k++){
            vector<double> h(n, 0);
            for(int i = 0;i<n;i++){
                for(int j = 1; j<m; j++)
                    h[i] += data[i][j]* weights[j];
                h[i] = 1.0 / (1 + exp(-h[i]));
                h[i] = is_treated[i] - h[i];
            }
            for(int i = 1;i<m;i++){
                delta[i] = 0;
                for(int j = 0;j < n;j++)
                    delta[i] += h[j]* data[j][i];
                weights[i] += delta[i] * alpha;
            }
        }
        //for(int i =1; i< m ;i++) cout<<weights[i] << " "<<endl;
        return weights;
    }

    double predict(const vector<double> & weights, const vector<double> & property)
    {
        double value = 0;
        for(int k = 1;k < weights.size();k++) value += property[k] * weights[k];
        value = 1.0/(1 + exp(-value));
        //cerr<< "predict "<<value<<endl; 
        return value;
    }
    void cross_validation_5_fold(std::vector< vector<double> >& data, const std::vector<bool> gound_truth)
    {
        int n = data.size(), m = data[0].size();
        int relevant1 = 0, relevant2  = 0, retrieved = 0, non_retrieved = 0;
        assert(n >= 5);
        for(int i = 0 ; i < 5; i++){
            std::vector<vector<double> > part, valid;
            std::vector<bool> part_label, valid_label;
            for(int j = 0; j < n; j++)
            {
                if(j >= n * i / 5 && j < n *(i + 1)/5) {valid.push_back(data[j]);valid_label.push_back(gound_truth[j]);}
                else {part.push_back(data[j]);part_label.push_back(gound_truth[j]);}
            }
            vector<double> weights = logistic_regression(part, part_label);
            double accuracy = 0;
            for(int j = 0;j < valid.size();j++)
            {
                double value = predict(weights, valid[j]);
                if(value > 0.5) {
                    retrieved++;
                    if(valid_label[j]) relevant1++;
                }else{
                    non_retrieved++;
                    if(!valid_label[j]) relevant2++;
                }
                //cerr<< "accuracy "<< valid_label[j] << value <<endl;
            }
        }
        information_retrieval(relevant1, relevant2, retrieved, non_retrieved);   
    }

    std::vector<double> ada_boosting(std::vector< std::vector<double> >& data, std::vector< std::vector<double> >& learner, const std::vector<bool> gound_truth)
    {
        int n = data.size(), m = data[0].size(), t = learner.size();
        std::vector<double> d(n, 1.0/n), alpha(n);
        for(int k = 0;k < learner.size();k++)
        {
            double err = 0;
            for(int i = 0;i < n; i++)
            {
                err += d[i] * ((predict(learner[k], data[i]) > 0.5) != gound_truth[i]);
            }
            alpha[k] = 0.5 * log((1 - err) / err);
            double sum  = 0;
            for(int i = 0;i < n; i++)
                d[i] = d[i] * exp(-alpha[k] * (((predict(learner[k], data[i]) > 0.5) != gound_truth[i])? -1 : 1)), sum += d[i];
            for(int i = 0;i < n; i++) // normalize
                d[i] /= sum;
        }
        return alpha;
        // std::vector<bool> ret(n);
        // for(int i = 0;i < n;i++){
        //     double ans = 0;
        //     for(int j = 0;j < t;j++)
        //         ans += alpha[j] * (predict(learner[j], data[i]) > 0.5 ? 1 : -1);
        //     ret[i] = ans < 0 ? false : true;
        // }
    }
    void information_retrieval(int relevant1, int relevant2, int retrieved, int non_retrieved){
        double precision, recall;
        cout << "precision: "<< (precision = relevant1 * 1.0 / retrieved) <<endl;
        cout << "recall: "<< (recall = relevant1 * 1.0 / (relevant1 + relevant2))<<endl;;
        cout << "fall-out: " << (retrieved - relevant1)* 1.0 / (retrieved + non_retrieved - relevant1 - relevant2)<<endl;
        cout << "F-score: " << 2.0 * precision * recall / (precision + recall)<<endl;
        cout<< "//-------------------------" <<endl;

    }
}