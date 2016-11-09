#include "line.h"
#include "word2vec.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <pthread.h>

using namespace std;
using namespace sae::io;

#define SIGMOID_BOUND 6
#define SIGMOID_TABLE_SIZE  2000
#define NEG_SAMPLING_POWER 0.75


LINE::LINE(MappedGraph *graph)
	:Solver(graph) {
}

LINE::~LINE() {
}

double init_alpha = 0.025, alpha;
int num_threads = 1, order = 2, dim = 100, num_negative = 5;
int num_vertex = 0, num_edges = 0;
long long total_samples = 1, current_sample_count = 0;

vector<double> degree;
vector<int> edge_source_id;
vector<int> edge_target_id;
vector<double> edge_weight;

vector<vector<double>> emb_vertex;
vector<vector<double>> emb_context;

// parameters for edge sampling
vector<vid_t> edge_table;
vector<double> edge_prob;
vector<vid_t> node_table;
vector<double> node_prob;

vector<double> sigmoid_table;


// read network from SAE graph
void ReadDataFromSAE(MappedGraph *graph)
{
	double weight = 1.0;
    num_edges = graph->EdgeCount();
	num_vertex = graph->VertexCount();
	edge_source_id = vector<int>(num_edges, 0);
	edge_target_id = vector<int>(num_edges, 0);
	edge_weight = vector<double>(num_edges, 0);
	degree = vector<double>(num_vertex, 0);;

	int k = 0;
	for (auto itr = graph->Edges(); itr->Alive(); itr->Next())
	{
	    vid_t x = itr->Source()->GlobalId(), y = itr->Target()->GlobalId();
	    degree[x] += weight;
	    edge_source_id[k] = x;
	    degree[y] += weight;
	    edge_target_id[k] = y;
	    edge_weight[k++] = weight;
	}
}

// update embeddings
void Update(vector<double> &vec_u, vector<double> &vec_v, vector<double> &vec_error, int label)
{
	double f = 0, g = 0;
	for (int j = 0; j < dim; j++)
		f += vec_u[j] * vec_v[j];
	if (f > SIGMOID_BOUND)
		g = (label - 1) * alpha;
	else if (f < -SIGMOID_BOUND)
		g = (label - 0) * alpha;
	else
		g = (label - sigmoid_table[floor((f + SIGMOID_BOUND) * SIGMOID_TABLE_SIZE / SIGMOID_BOUND / 2)]) * alpha;
	for (int j = 0; j < dim; j++)
		vec_error[j] += g * vec_v[j];
	for (int j = 0; j < dim; j++)
		vec_v[j] += g * vec_u[j];
}

void *TrainLINEThread(void *id)
{
	long long count = 0, last_count = 0;
	vector<double> vec_error(dim, 0);
	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count>10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%calpha: %f  Progress: %.3lf%%", 13, alpha, (double)current_sample_count / (double)(total_samples + 1) * 100);
			fflush(stdout);
			alpha = init_alpha * (1 - current_sample_count / (double)(total_samples + 1));
			if (alpha < init_alpha * 0.0001)
				alpha = init_alpha * 0.0001;
		}

		vid_t rand_edge = Word2Vec::alias_draw(edge_table, edge_prob);
		vid_t u = edge_source_id[rand_edge];
		vid_t v = edge_target_id[rand_edge];
		for (int j = 0; j < dim; j++)
			vec_error[j] = 0;

		//negative sampling
		int label = 1 ;
		vid_t target = v;
		for (int k = 0; k < num_negative ; k++)
		{
			if (k == 0)
			{
				target = v;
				label = 1;
			}
			else
			{
				target = Word2Vec::alias_draw(node_table, node_prob);
				label = 0;
			}
			if (order == 1)
				Update(emb_vertex[u], emb_vertex[target], vec_error, label);
			if (order == 2)
				Update(emb_vertex[u], emb_context[target], vec_error, label);
		}
		// update vertex embedding
		for (int j = 0; j < dim; j++)
			emb_vertex[u][j] += vec_error[j];
		count++;
	}
}


void TrainLINE(MappedGraph* graph)
{
    ReadDataFromSAE(graph);
	//initialize the vertex embedding and the context embedding
	emb_vertex = vector<vector<double>> (num_vertex, vector<double>(dim, 0));
	emb_context = vector<vector<double>> (num_vertex, vector<double>(dim, 0));
	for (int i = 0; i < num_vertex; i++)
		for (int j = 0; j < dim; j++)
			emb_vertex[i][j] = (rand() / (double)RAND_MAX - 0.5) / dim;

	//set up edges sampling table with alias sampling according to weight
	edge_table = vector<vid_t> (num_edges, 0);
	edge_prob = vector<double> (num_edges, 0);
	double sum = accumulate(edge_weight.begin(), edge_weight.end(), 0);
	for (int k = 0; k < num_edges; k++)
		edge_prob[k] = edge_weight[k] / sum;
	edge_table = Word2Vec::alias_setup(edge_prob);

	//set up verte sampling table with alias sampling according to degrees
	node_prob = vector<double>(num_vertex, 0);
	for (int k = 0; k < num_vertex; k++)
		node_prob[k] = pow(degree[k], NEG_SAMPLING_POWER);
	sum = accumulate(node_prob.begin(), node_prob.end(), 0);
	for (int k = 0; k < num_vertex; k++)
		node_prob[k] /= sum;
	node_table = Word2Vec::alias_setup(node_prob);

	//set up sigmoid table
	sigmoid_table = vector<double> (SIGMOID_TABLE_SIZE + 1, 0);
	for (int i = 0; i < SIGMOID_TABLE_SIZE; i++)
    {
        sigmoid_table[i] = exp((i * 1.0 / SIGMOID_TABLE_SIZE * 2 - 1) * SIGMOID_BOUND);
        sigmoid_table[i] = sigmoid_table[i] / (sigmoid_table[i] + 1);
    }

	//set up multi-thread
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (int i = 0; i < num_threads; i++)
        pthread_create(&pt[i], NULL, TrainLINEThread, (void *)&i);
    for (int i = 0; i < num_threads; i++)
        pthread_join(pt[i], NULL);
}


vector<vector<double> > MergeVector(vector<vector<double>>& vec1, vector<vector<double>>&vec2)
{
	int dim1 = vec1[0].size(), dim2 = vec2[0].size();
    vector<vector< double> > ret(num_vertex);
	for (int i = 0; i < num_vertex; i++)
    {
		double len = 0;
		for (int j = 0; j < dim1; j++)
            len += vec1[i][j] * vec1[i][j];
		len = sqrt(len);
		for (int j = 0; j < dim1; j++)
            vec1[i][j] /= len;

		len = 0;
		for (int j = 0; j < dim2; j++)
            len += vec2[i][j] * vec2[i][j];
		len = sqrt(len);
		for (int j = 0; j < dim2; j++)
            vec2[i][j] /= len;

        for (int j = 0; j < dim1; j++)
            ret[i].push_back(vec1[i][j]);
		for (int j = 0; j < dim2; j++)
            ret[i].push_back(vec2[i][j]);
	}
    return ret;
}


vector<vector<double> >  LINE::solve(int T, int d, int Th, int Neg, double init_rate) {

    init_alpha = init_rate;
	alpha = init_alpha;
	
	total_samples = T;
	dim = d;
	num_negative = Neg;
	num_threads = Th;

	printf("Samples: %lld\n", total_samples);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial alpha: %lf\n", init_alpha);

	order = 2;
    TrainLINE(graph);
    printf("Second-order proximity training completed\n");
    vector<vector<double> > vec1 = emb_vertex;

    order = 1;
	alpha = init_alpha;
	current_sample_count = 0;
    TrainLINE(graph);
    printf("First-order proximity training completed\n");
    vector<vector<double> >vec2 = emb_vertex;

    vector<vector<double> > ans =  MergeVector(vec1, vec2);
	return ans;
}
