#include "argparse/macro-argparse-jquery.hh"
#include "solver/solver.h"
#include "solver/solverForStreaming.h"
#include "sampling/triangle_sampling.h"
#include "sampling/community_detection_sampling.h"
#include "basic/triangle_count.h"
#include "basic/simrank.h"
#include "basic/pagerank.h"
#include "basic/degree_distribution.h"
#include "basic/community_detection.h"
#include "basic/shortest_path.h"
#include "basic/propensity_score_matching.h"
#include "basic/k_core_decomposition.h"
#include "storage/graph_builder.h"
#include "storage/mgraph.h"
#include "report/table_generator.h"
#include "influence/influence_maximization.h"
#include "social_analysis/social_main.h"
#include "streaming/dynamicMinimumSpanningTree.h"
#include "network_embedding/deepwalk.h"
#include "network_embedding/node2vec.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <map>
#include <set>
#include <cstdlib>
#include <cstring>


using namespace std;
using namespace sae::io;

DEF_ARGUMENT_CLASS(
	Argument,
	std::string,	input,		"",				REQUIRED,   OPT_SLH(-i, --input, "input data"),
    std::string,    output,     "./output",     OPTIONAL,   OPT_SLH(-o, --output, "output direction"),
    std::string,    query,     "",     OPTIONAL,   OPT_SLH(-q, --query, "query file "),
    std::string,    task,       "",             OPTIONAL,   OPT_SLH(-t, --task, "declear task:im(inf-max), dd(degree), pr(PageRank), tr(triangle), sp(ShortestPath), social(social analysis)\n"),
    int,            para_im_k,  "2",            OPTIONAL,   OPT_SLH(-k, --seed, "seed size"),
    std::string,    para_edge_w,    "rand",     OPTIONAL,   OPT_SLH(-w, --weight, "edge weights:rand,const,deg"),
    double,         para_const, 0.8,            OPTIONAL,   OPT_SLH(-c, -constant, "constant value"),
    int,            para_start, "0",            OPTIONAL,   OPT_SLH(-s, --start, "start point"),
    int,            para_end, "1",            OPTIONAL,   OPT_SLH(-e, --end, "end point"),
    int,            para_cd_task, 1,            OPTIONAL,   OPT_SLH(-r, --run, "run community detection sub task"),
    double,            para_sample_probability, 0.01,            OPTIONAL,   OPT_SLH(-p, --probability, "sample probability for aglorithm")
);

string output_dir, output_name;

void makeFakeData(int numVertex=10, double p = 0.1, int properties_num = 3) {
    //int numEdge = rand() % (numVertex * numVertex / 3) + numVertex;
    int numEdge = (numVertex * (numVertex - 1)) / 2 * p;
    GraphBuilder<int> graph;
    vector< vector<double> > all_properties(numVertex, std::vector<double>(properties_num));
    map<pair<int, int>, bool> edges;
    for (int i = 0; i < numEdge; ++i) {
        pair<int, int> edge;
        do {
            int x = rand() % (numVertex - 1);
            int y = rand() % (numVertex - x - 1) + x + 1;
            edge = make_pair(x, y);
        } while (edges.find(edge) != edges.end());
        edges[edge] = true;
    }
    for (int i = 0; i < numVertex; ++i)
    {
        for(int j = 1;j < properties_num; j++)
            all_properties[i][j] = (rand() % 100) / 100.0;
    }
    for(int i = 0; i < numVertex; i++)
        {
            for(int j = 0; j < numVertex; j++)
                    if(edges.find(make_pair(i,j)) != edges.end() || edges.find(make_pair(j,i)) != edges.end() )
                    {
                        //balance properties of i,j
                        double & x = all_properties[i][1];
                        double & y = all_properties[j][1];
                        double xx = x, yy = y;
                        double alpha = 0.4;
                        x = alpha * xx + (1 - alpha) * yy;
                        y = (1 - alpha) * xx + alpha * yy;
                    }
        }
    for(int i = 0; i < numVertex; i++){
        vector<double>& properties = all_properties[i];
        //homophily's influence
        int possibility = (int)(properties[1] * 20) + 1;
        if(rand() % possibility == 0) properties[0] = 1;
            else properties[0] = -1;
    }
        for(int i = 0; i < numVertex; i++)
        if(all_properties[i][0] > 0)
        {
            // opinion leader's influence
            for(int j = 0; j < numVertex; j++)
                if(all_properties[j][0] < 0)
                    if(edges.find(make_pair(i,j)) != edges.end() || edges.find(make_pair(j,i)) != edges.end() )
                    {
                        if(rand() % 600 == 0) all_properties[j][0] = 1;
                    }
        }
    for(int i = 0;i< numVertex; i++)
    {
        graph.AddVertex(i, all_properties[i]);
        //cout<<"v: id "<<i<<" "<<all_properties[i][0] <<" "<< all_properties[i][1]<<endl;
    }
    for(int i = 0; i< numVertex; i++)
        for(int j = i + 1; j< numVertex; j++)
        if(edges.find(make_pair(i, j)) != edges.end())
        {
            pair<int ,int > edge = make_pair(i, j);
            int value = rand() % 50;
            graph.AddEdge(edge.first, edge.second, value);
            graph.AddEdge(edge.second, edge.first, value);
            //cout << edge.first << " " << edge.second << endl;
        }
    system("mkdir -p fake");
    graph.Save("./fake/graph");
}



map<string, int> nodeMap;
map<vid_t, vid_t> mapToReal;
map<vid_t, vid_t> mapToSae;

int GetOrInsert(const string& key)
{
    map<string, int>::iterator it = nodeMap.find(key);
    if (it != nodeMap.end())
        return it -> second;
    int id = (int) nodeMap.size();
    int real_id=atoi(key.c_str());
    nodeMap.insert(make_pair(key, id));
    mapToReal.insert(make_pair(id, real_id));
    return id;
}

int makeData(string input,string output) {
    GraphBuilder<int> graph;
    ifstream fin(input.c_str());
    //ifstream fin("./resource/twitter_combined.txt");
    string buf;
    //for (int i = 0; i < 4; ++i) getline(fin, buf);
    int v_cnt(-1);
    map<string, int> nodeMap;
    while (1) {
        string x, y;
        if (!(fin >> x >> y)) break;
        int a = GetOrInsert(x);
        int b = GetOrInsert(y);
        int c = max(max(v_cnt, a), b);
        while (v_cnt < c) graph.AddVertex(++v_cnt, 0);
        graph.AddEdge(a, b, 0);
        graph.AddEdge(b, a, 0);
    }
    cout << graph.VertexCount() << " " << graph.EdgeCount() << endl;
    graph.Save((output).c_str());
    FILE* fout = fopen(( output+"_map").c_str(), "w");
    map<vid_t, vid_t>::iterator it;
    for(it=mapToReal.begin();it!=mapToReal.end();++it)
    {
        fprintf(fout,"%llu\t%llu\n",it->first,it->second);
    }
    fclose(fout);
    return 0;
}

void readNodeMap(string input)
{
    mapToReal.clear();
    mapToSae.clear();
    ifstream fin((input+"_map").c_str());
    vid_t x,y;
    while(1){
        if (!(fin >> x >> y)) break;
        mapToReal.insert(make_pair(x,y));
         mapToSae.insert(make_pair(y,x));
    }
}


int makeDataForStreaming(){
	ifstream fin("./resource/facebook.txt");
    long long n=0,m=0,x,y;
    while (fin >> x >> y)
    {
        ++m;
        if (x>n) n=x;
        if (y>n) n=y;
    }
    ++n;
    ofstream fout("./data/facebook.txt");
    fout<<n<<' '<<m<<endl;
    ifstream fin2("./resource/facebook.txt");
    while (fin2 >> x >> y)
    {
        ++m;fout<<x<<' '<<y<<' '<<rand()%1000+1<<endl;
    }
    return 0;
}

int makeFakeDataForStreaming(){
	int n,m,i;
	n=10000;m=1000000;
	ofstream fout("./fake/graph.txt");
	fout<<n<<' '<<m<<endl;
	for (i=1;i<=m;++i)
		fout<<rand()%n<<' '<<rand()%n<<' '<<rand()%1000+1<<endl;
    return 0;
}

void runPageRank(MappedGraph *graph) {
    PageRank pr(graph);
    time_t start_time = clock();
    vector<pair<sae::io::vid_t, double>> res = pr.solve();
    FILE *fout = fopen((output_dir + "/pagerank").c_str(), "w");
    fprintf(fout, "vid\tpage_rank_score\n");
    for (unsigned int i = 0; i < res.size(); i ++)
        fprintf(fout, "%llu\t%.5lf\n", res[i].first, res[i].second);
    fclose(fout);
    time_t end_time = clock();
    cout << "Running time of PageRank: " << (end_time - start_time + 0.0) / CLOCKS_PER_SEC << endl;
}

void runDegreeDistribution(MappedGraph *graph) {
    Degree_Distribution dd(graph);
    time_t start_time = clock();
    vector<pair<int, double>> res = dd.solve();
    FILE *fout = fopen((output_dir + "/degree_distribution").c_str(), "w");
    fprintf(fout, "degree\tproportion\n");
    for (unsigned int i = 0; i < res.size(); i ++) {
        fprintf(fout, "%d\t%.5lf\n", res[i].first, res[i].second);
    }
    fclose(fout);
    fout = fopen((output_dir + "/degree_of_each_vertex").c_str(), "w");
    for (auto iter = graph -> Vertices(); iter -> Alive(); iter -> Next()) {
        fprintf(fout, "%llu\t%d\n", iter -> GlobalId(), iter -> OutEdgeCount());
    }
    fclose(fout);
    time_t end_time = clock();
    cout << "Running time of Degree_Distribution: " << (end_time - start_time + 0.0) / CLOCKS_PER_SEC << endl;
}

void runInfluenceMaximization(MappedGraph *graph, int K, double W) {
    time_t start_time = clock();
    Influence_Maximization im(graph);
    pair<vector<sae::io::vid_t>, double> res = im.solve(K, 20, W);
    FILE* fout = fopen((output_dir + "/influence_maximization").c_str(), "w");
    fprintf(fout, "Expected #actived users: %.5lf\n", res.second);
    cout << "Expected #actived users: " << res.second << endl;
    for (unsigned int i = 0; i < res.first.size(); i ++) {
        fprintf(fout, "%llu\n", res.first[i]);
    }
    fclose(fout);
    time_t end_time = clock();
    cout << "Running time of Influence_Maximization: " << (end_time - start_time + 0.0) / CLOCKS_PER_SEC << endl;
}

void runShortestPath(MappedGraph *graph, long long start, long long end, bool require_path = false)
{
    time_t start_time = clock();
    Shortest_Path sp(graph);
    std::vector<sae::io::EdgeIteratorPtr> ans;
    double len = sp.solve(start,end,ans);
    if(len < 0){
        cout<<"There is no path between "<<start<<" and "<<end<<endl;
        return;
    }
    cout<<" Length of Shortest Path between "<<start<<" and "<<end<<" is "<<len<<endl;
    if(require_path){
        cout<<"Path: "<<endl<<start<<endl;
        for(auto iter = ans.begin();iter != ans.end();iter++)
            cout<<(*iter)->TargetId()<<endl;
    }
    time_t end_time = clock();
    cout << "Running time of Shortest_Path: " << (end_time - start_time + 0.0) / CLOCKS_PER_SEC << endl;
}
void runCommunityDetection(MappedGraph *graph,string input,int sub_task,int K)
{
    cout<<"\tRun community detection algorithm"<<endl<<endl;
    time_t start_time = clock();
    Community_detection cd(graph);
    pair<vector<vid_t>,double> ans=cd.solve(K,sub_task);
    time_t end_time = clock();

    printf( "\tmodularity is %.4f\n",ans.second);
    printf( "\tRunning time of Community detection: %.4f\n",(end_time - start_time + 0.0) / CLOCKS_PER_SEC );

	FILE* fout = fopen((  output_dir+"/community_detection").c_str(), "w");
	fprintf(fout, "modularity is %.4f\nvertex_id\tcommunity_id\n",ans.second);
	readNodeMap(input);
	for(unsigned int i=0;i<ans.first.size();i++)
        fprintf(fout, "%llu\t%llu\n",mapToReal[i],ans.first[i]);
	fclose(fout);
}

void runCommunityDetectionSampling(MappedGraph *graph,string input,int sub_task,double p,int K,int sm,int e)
{
    cout<<"\tRun community detection sampling algorithm"<<endl<<endl;
	FILE* fout = fopen((  output_dir+"/community_detection_sampling").c_str(), "w");
    time_t start_time = clock();
    Community_detection_sampling cd(graph);
    //cd.test_community_sampling(graph,10,90,2,4);
    pair<vector<vid_t>,double> ans=cd.solve(p,K,sub_task,sm,e);
	fprintf(fout, "modularity is %.4f\nvertex_id\tcommunity_id\n",ans.second);
	readNodeMap(input);
	for(unsigned int i=0;i<ans.first.size();i++)
        fprintf(fout, "%llu\t%llu\n",mapToReal[i],ans.first[i]);
    time_t end_time = clock();
    printf( "\tmodularity is %.4f\n\n",ans.second);
	cout<< "\tRunning time of Community detection: "<<((end_time - start_time + 0.0) / CLOCKS_PER_SEC )<<endl;
	fclose(fout);
}

void testCommunityDetectionSampling(MappedGraph *graph,string input,int sub_task,int sm,int e)
{
    cout<<"\tTestcommunity detection sampling algorithm"<<endl<<endl;
    Community_detection_sampling cd(graph);
    cd.test_community_sampling(graph,sub_task,sm,e);

}
void runKCoreDecomposition(MappedGraph *graph)
{
    cout<<"\tRun k-core decomposition algorithm"<<endl<<endl;
    time_t start_time = clock();
    K_Core_Decomposition cd(graph);
    vector<pair<vid_t, vid_t> > ans=cd.solve();
    vid_t n=ans.size();
    cout<<"No.  |  k-shell:"<<endl;
    for (vid_t i=0;i<n;++i)
    	cout<<ans[i].first<<' '<<ans[i].second<<endl;
    time_t end_time = clock();
    cout << "Running time of k-core decomposition: " << (end_time - start_time + 0.0) / CLOCKS_PER_SEC << endl;
}
void makeTencentData(string input ,string output)
{
    //input:    /tmp/tencent8.graph     output: ./data/tencent_weibo
    ifstream fin(input);
    int n,m;
    GraphBuilder<int> graph;
    fin>>n>>m;
    for(int i = 0;i<n;i++)
    {
		mapToReal.insert(make_pair(i, i));
        graph.AddVertex(i,0);
        if(i % 1000 == 0) cerr<<i<<" / "<<n<<endl;
    }
    cerr<<"finish vertices"<<endl;
    for(int i = 0;i<m;i++)
    {
        int x,y,not_use;
        if(!(fin >> x >> y >> not_use)) {
            cerr<<"number of edges does not tally"<<endl;
            break;
        }
        graph.AddEdge(x,y,0);
        graph.AddEdge(y,x,0);
        if(i % 10000 == 0)
            cerr<<i<<" / "<<m<<endl;
    }
    graph.Save((output_dir).c_str());
	FILE* fout = fopen(( output+"_map").c_str(), "w");
    map<vid_t, vid_t>::iterator it;
    for(it=mapToReal.begin();it!=mapToReal.end();++it)
    {
        fprintf(fout,"%llu\t%llu\n",it->first,it->second);
    }
    fclose(fout);
}

void makeTencentDataForStreaming()
{
    ifstream fin("/tmp/tencent8.graph");
    ofstream fout("./output/tencent8AddRandomWeight.graph");
    int n,m,i,x,y,w;
    fin>>n>>m;
    fout<<n<<' '<<m<<endl;
    for (i=1;i<=m;++i)
    {
	if (i%1000000==0) cout<<i<<endl;
        fin>>x>>y>>w;
        fout<<x<<' '<<y<<' '<<rand()%65536*1.0/65536<<endl;
    }
    fin.close();fout.close();
}

void makeExpertData()
{
    GraphBuilder<int> graph;
    ifstream fin("./resource/expert.txt");
    //ifstream fin("./resource/twitter_combined.txt");
    string buf;
    //for (int i = 0; i < 4; ++i) getline(fin, buf);
    int v_sum = -1;
    std::vector<int> v_cnt(100000,0);
    set<pair<int, int> >edges;
    while(1)
    {
        int num;
        if(!(fin >> num)) break;
        vector<int> points;
        for(int i = 0; i<num;i++) {
            int tmp = 0;
            fin >> tmp;
            points.push_back(tmp);
            v_cnt[tmp]++;
            while(v_sum < tmp) graph.AddVertex(++v_sum,0);
            for(int j = 0; j<i; j++)
                if(edges.find(make_pair(points[j],points[i])) == edges.end())
                {
                    graph.AddEdge(points[j],points[i],0);
                    graph.AddEdge(points[i],points[j],0);
                    edges.insert(make_pair(points[j],points[i]));
                }
        }
        cerr<< "v_sum: "<<v_sum<<endl;
        if(v_sum > 20000) break;
    }
    cout << graph.VertexCount() << " " << graph.EdgeCount() << endl;
    graph.Save("./data/expert");
    MappedGraph* mgraph = MappedGraph::Open("./data/expert");

    vector< vector<double> > data;

    Social_Solver solver(mgraph);
    vector<int> degrees = solver.Degree_Centrality();

    vector<pair<double, int> >betweenness = solver.Unweighted_Betweenness();
    vector<double> effective  = solver.Effective_Size();
    std::vector<int> kcore = solver.K_Core();
    for(int i = 0;i< mgraph->VertexCount();i++)
    {
        std::vector<double> property;
        property.push_back(v_cnt[i]);
        if(v_cnt[i] >= 3) cerr << i <<endl;
        property.push_back(degrees[i]);
       // property.push_back(effective[i]);
        property.push_back(kcore[i]);
        data.push_back(property);
    }
    for(int i = 0;i < betweenness.size(); i++)
        data[betweenness[i].second].push_back(betweenness[i].first);
    for(int i = 0; i< mgraph->VertexCount();i++)
        graph.AddVertex(i, data[i]);
    graph.Save("./data/expert");
}
void runExpertClassification(MappedGraph* graph)
{
    cout << "Please input the path of ground_truth file: "<<endl;
    string ground_truth_file;
    cin >> ground_truth_file;
    time_t start_time = clock();
    ifstream fin(ground_truth_file.c_str());
    int t;
    vector<int> experts;
    std::vector<bool> ground_truth(graph->VertexCount(), false);
    while(fin >> t){
        if(t <= graph->VertexCount())
        {
            ground_truth[t] = true;
            experts.push_back(t);
        }

    }
    std::vector< vector<double> > data, data2;
    vector<bool> ground_truth2;
    auto viter = graph->Vertices();

    for(int i = 0;i<graph->VertexCount();i++)
    {
        viter->MoveTo(i);
        data.push_back(sae::serialization::convert_from_string< vector<double> >(viter -> Data()) );
        data[i].push_back(data[i][0]);
    }

    statistic::cross_validation_5_fold(data, ground_truth);

    int set_number = 12, sample_number = 11;
    std::vector< std::vector<double> > learner;
    vector< std::vector<double> > parts_data;
    std::vector<bool> parts_truth;
    std::vector<bool> used(graph->VertexCount(), false);
    for(int k = 0; k < set_number; k++)
    {
        //cout<< "new sample"<<endl;
        vector< std::vector<double> > part_data;
        std::vector<bool> part_truth;
        for(int i = 0;i<sample_number;i++)
        {
            int r1 = rand() % experts.size(), r2 = rand() % data.size();
            part_data.push_back(data[experts[r1]]);
            part_truth.push_back(ground_truth[experts[r1]]);
            part_data.push_back(data[r2]);
            part_truth.push_back(ground_truth[r2]);
            if(!used[experts[r1]]){
                used[experts[r1]] = true;
                parts_data.push_back(data[experts[r1]]);
                parts_truth.push_back(ground_truth[experts[r1]]);
            }
            if(!used[r2]){
                used[r2] = true;
                parts_data.push_back(data[r2]);
                parts_truth.push_back(ground_truth[r2]);
            }
        }
        learner.push_back(statistic::logistic_regression(part_data,part_truth));
        int tmp = 0;
        for(int i = 0; i < part_truth.size(); i++)
            tmp += (statistic::predict(learner[k], part_data[i])>0.5) == part_truth[i];
        //cerr<< "part accuracy: "<< tmp * 1.0/ part_truth.size() <<endl;
     }
    //data = parts_data;
    //ground_truth = parts_truth;
    vector<double> alpha = statistic::ada_boosting(parts_data, learner, parts_truth);
    std::vector<bool> result(data.size());
    for(int i = 0;i < data.size();i++){
            double ans = 0;
            for(int j = 0;j < learner.size();j++)
                ans += alpha[j] * (statistic::predict(learner[j], data[i]) > 0.5 ? 1 : -1);
            result[i] = ans < 0 ? false : true;
    }
    int tmp = 0, relevant1 = 0, relevant2  = 0, sum1 = 0, sum2 = 0, retrieved = 0, non_retrieved = 0;
    for(int i = 0; i < ground_truth.size(); i++)
    {
        if(result[i] == ground_truth[i]){
            tmp ++;
            if(ground_truth[i]) relevant1++; else relevant2 ++;
        }
        if(ground_truth[i]) sum1 ++; else sum2 ++;
        if(result[i]) retrieved++; else non_retrieved++;
    }
    statistic::information_retrieval(relevant1, relevant2, retrieved, non_retrieved);
    time_t end_time = clock();
    cout<<"Running time of Expert Classification algorithm: " << (end_time - start_time + 0.0)/ CLOCKS_PER_SEC <<'s'<<endl;
}

void runPropensityScoreMatching(MappedGraph* graph){
    Propensity_Score_Matching  psm(graph);
    output_name = "Propensity Score Matching.txt";
    ofstream fout((output_dir + output_name).c_str());
    cout << "Please input the determine_coefficient: "<<endl;
    int determine_coefficient = 0;
    cin >> determine_coefficient;
    time_t start_time = clock();
    vector<int> nodes = psm.solve(determine_coefficient);
    time_t end_time = clock();
    for(int i = 0 ;i < nodes.size(); i += 2)
    {
        fout << nodes[i] << "---" << nodes[i + 1] <<endl;
    }
    cout<<"Running time of Propensity_Score_Matching algorithm: " << (end_time - start_time + 0.0)/ CLOCKS_PER_SEC <<'s'<<endl;
}

void runDynamicMinimumSpanningTree(int mode,string file_path)
{
    cout<<"\tRun dynamic minimum spanning tree algorithm"<<endl<<endl;
    time_t start_time = clock();
    dynamicMinimumSpanningTree cd(file_path);
    resultMST* ans;
    if (mode==0)
	ans=cd.solve();
    if (mode==1)
	ans=cd.solve_raw(1);
    if (mode==2)
	ans=cd.solve_raw(0);
    vid_t n=ans->edge.size(),i;
    cout<<"algorithm done. Writing results at './output/dynamicMST.txt' ..."<<endl;
    ofstream fout("./output/dynamicMST.txt");
    time_t end_time = clock();
    fout<<"Running time of dynamic MST algorithm: " << (end_time-start_time+0.0)/ CLOCKS_PER_SEC <<'s'<<endl;
    fout<<"nodes:"<<ans->n<<",edges:"<<ans->m<<"\nmstValue:"<<ans->mstValue<<"\nvertex1 vertex2 weight:\n";
    for (i=0;i<n;++i) if (ans->edge[i].first.first!=-1)
    {
    	if (i%100000==0) cout<<i<<endl;
        fout<<ans->edge[i].first.first<<' '<<ans->edge[i].first.second<<' '<<ans->edge[i].second<<endl;
    }
    end_time = clock();
    fout << "Running time of dynamic minimum spanning tree: " << (end_time - start_time + 0.0) / CLOCKS_PER_SEC << "s"<<endl;
    fout.close();
}


void runSimRank(MappedGraph *graph,string input,string query,int sub_task,double c,vid_t v,int K)
{

    cout<<"\tRun simrank algorithm"<<endl<<endl;
	FILE* fout = fopen((  output_dir+"/simrank").c_str(), "w");
    bool is_accurate=(sub_task==0);
    SimRank sr(graph);
    time_t start_time = clock();
    readNodeMap(input);
    vector<vid_t> query_list;
    vector<vid_t> query_list_real;
    if(query!="")
    {
        ifstream fin(query.c_str());
        vid_t id;
        while(1)
        {
            if(!(fin>>id)) break;
            query_list.push_back(mapToSae[id]);
            query_list_real.push_back(id);
        }
    }
    else
    {
        query_list.push_back(mapToSae[v]);
        query_list_real.push_back(v);
    }

    vector<vector<pair<double, vid_t> >> ans=sr.solve(query_list,is_accurate,c);
    time_t end_time = clock();
    for(int j=0;j<query_list.size();j++)
    {
        fprintf(fout,"%llu\t%d\n",query_list_real[j],K);
        for (int i=0;i<K;i++)
            fprintf(fout,"%llu\t%f\n",mapToReal[ans[j][i].second],ans[j][i].first);
        fprintf(fout,"\n\n");
    }

    printf( "Running time of simrank algorithm: %.4f\n",(end_time - start_time + 0.0) / CLOCKS_PER_SEC);
}

void runDeepWalk(MappedGraph *graph,string input,string output)
{
    cout<<"Run deepwalk algorithm"<<endl<<endl;
    time_t start_time = clock();
    Deep_Walk dw(graph);
    vector<vector<double> >  ans = dw.solve(5,100,100,10);
    time_t end_time = clock();
    printf( "Running time of word2vec algorithm: %.4f\n",(end_time - start_time + 0.0) / CLOCKS_PER_SEC);
}

void runNode2vec(MappedGraph *graph,string input,string output)
{
    cout<<"Run node2vec algorithm"<<endl<<endl;
    time_t start_time = clock();
    Node2Vec nv(graph);
    vector<vector<double> >  ans = nv.solve(10,100,50,5);
    time_t end_time = clock();
    printf( "Running time of word2vec algorithm: %.4f\n",(end_time - start_time + 0.0) / CLOCKS_PER_SEC);
}

int main(int argc, char **argv) {
    int vertexNum = 40;
    double edgeProb = 0.8;
    srand(time(NULL));

    // makeFakeData(vertexNum, edgeProb);
    // return 0;
//	makeFakeDataForStreaming();
//	makeTencentDataForStreaming();
//	makeDataForStreaming();
//  makeFakeData(vertexNum, edgeProb);
 // makeExpertData();
//    makeTencentData();
//    return 0;
	// parse arguments
	Argument args;
	if (! args.parse_args(argc, argv))
        return 1;
    // declare input file
    if (args.input().length() == 0)
        return 1;

    string task = args.task();
    string input= args.input();
    string query=args.query();
    printf("Input: %s\n", input.c_str());
    // declare output file direction
    if (args.output().length() > 0) {
        output_dir = args.output().c_str();
    }
	//if(!(task=="md"||task=="mt"))
	//	system(("mkdir -p " + output_dir).c_str());
    printf("Output: %s\n", output_dir.c_str());
    double p=args.para_sample_probability();
    double c = args.para_const();
    int sub_task=args.para_cd_task();
    int K =args.para_im_k();
    int s=args.para_start();
    int e=args.para_end();
    // generate a graph
    if (task == "gg") {
        makeFakeData(vertexNum, edgeProb);
        cout << "generate success!" << endl;
        return 0;
    }
    if (task == "md") {
        makeData(input,output_dir);
        cout << "generate success!" << endl;
        return 0;
    }

    if(task=="mt"){
        makeTencentData(input,output_dir);
        cout << "generate success!" << endl;
        return 0;
    }

    if (task =="dm"){
        runDynamicMinimumSpanningTree(0,args.input());
	    return 0;
	}

    if (task=="dmraw"){
	runDynamicMinimumSpanningTree(1,args.input());
	return 0;
    }

    if (task=="dmrawnw"){
	runDynamicMinimumSpanningTree(2,args.input());
	return 0;
    }

    MappedGraph *graph = MappedGraph::Open(args.input().c_str());
    cout << "===== graph information =====" << endl;
    cout << "#vertices: " << graph -> VertexCount() << endl;
    cout << "#edges: " << graph -> EdgeCount() << endl;
    cout << "=============================" << endl;

    if (task =="cd"){
	    runCommunityDetection(graph,input,sub_task,K);
    }

    if (task =="cs"){
	    runCommunityDetectionSampling(graph,input,sub_task,p,K,s,e);
    }

    if (task =="ct"){
	    testCommunityDetectionSampling(graph,input,sub_task,s,e);
    }

    if (task =="sr"){
        runSimRank(graph,input,query,sub_task,c,s,K);
    }


    if (task =="dw"){
        runDeepWalk(graph,input,output_dir);
    }
    if (task =="nv"){
        runNode2vec(graph,input,output_dir);
    }
    //declare task

    // call influence maximization
    if (task == "im") {
        int k = args.para_im_k();
        if (k <= 0) {
            cout << "The number of seeds must greater than zero!" << endl;
            return 1;
        }
        string weight_set = args.para_edge_w();
        if (weight_set == "const") {
            double w = args.para_const();
            if (w == 0) {
                cout << "Please define a constant value greater than zero!" << endl;
                return 1;
            }
            cout << "Running influence maximization with #seeds as " << k << endl;
            runInfluenceMaximization(graph, k, w);
        }
        if (weight_set == "rand") {
            cout << "Running influence maximization with #seeds as " << k << endl;
            runInfluenceMaximization(graph, k, 0.0);
        }
        if (weight_set == "deg") {
            cout << "Running influence maximization with #seeds as " << k << endl;
            runInfluenceMaximization(graph, k, -1.0);
        }
    }

    // call PageRank
    if (task == "pr") {
        runPageRank(graph);
    }

    // call degree distribution
    if (task == "dd") {
        runDegreeDistribution(graph);
    }
	if (task == "sp") {
        runShortestPath(graph,args.para_start(),args.para_end(),true);
    }
    if (task == "social"){
        output_name = "social_analysis.txt";
        ofstream fout((output_dir + output_name).c_str());
        social_main(graph, fout);
    }
    if (task == "ec"){
        runExpertClassification(graph);
    }

    if (task =="kc"){
	    runKCoreDecomposition(graph);
	}

    if (task == "psm"){
        runPropensityScoreMatching(graph);
    }

	//testTable();
	//makeFakeData(vertexNum, edgeProb);
	//testTriangle();

    return 0;
}
