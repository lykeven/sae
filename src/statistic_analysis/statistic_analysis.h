#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
namespace statistic{
    std::vector<double> logistic_regression
        (const std::vector< std::vector<double> >& data, const std::vector<bool> is_treated);
    double predict(const std::vector<double> & weights, const std::vector<double> & property);

    void cross_validation_5_fold(std::vector< std::vector<double> >& data, const std::vector<bool> gound_truth);
    void information_retrieval(int relevant1, int relevant2, int retrieved, int non_retrieved);

    std::vector<double> ada_boosting(std::vector< std::vector<double> >& data, std::vector< std::vector<double> >& learner, const std::vector<bool> gound_truth);
}