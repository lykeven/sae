#include "line.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <boost/thread.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
// Sun CC doesn't handle boost::iterator_adaptor yet
#if !defined(__SUNPRO_CC) || (__SUNPRO_CC > 0x530)
#include <boost/generator_iterator.hpp>
#endif

using namespace sae::io;

#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;



LINE::LINE(MappedGraph *graph)
	:Solver(graph) {
}

LINE::~LINE() {
}


typedef double real;                    // Precision of float numbers

struct ClassVertex {
	double degree;
	char *name;
};

char network_file[MAX_STRING], embedding_file[MAX_STRING];
struct ClassVertex *vertex;
int is_binary = 0, num_threads = 1, order = 2, dim = 100, num_negative = 5;
int *vertex_hash_table, *neg_table;
int max_num_vertices = 1000, num_vertices = 0;
long long total_samples = 1, current_sample_count = 0, num_edges = 0;
real init_rho = 0.025, rho;
real *emb_vertex, *emb_context, *sigmoid_table;

int *edge_source_id, *edge_target_id;
double *edge_weight;

// Parameters for edge sampling
long long *alias;
double *prob;

//random generator 
typedef boost::minstd_rand base_generator_type;
base_generator_type generator(42u);
boost::uniform_real<> uni_dist(0, 1);
boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);

char vector_file1[MAX_STRING], vector_file2[MAX_STRING], output_file[MAX_STRING];
int binary = 0;
long long vector_dim1, vector_dim2;
real *vec1, *vec2;


/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key)
{
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key)
	{
		hash = hash * seed + (*key++);
	}
	return hash % hash_table_size;
}

void InitHashTable()
{
	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;
}

void InsertHashTable(char *key, int value)
{
	int addr = Hash(key);
	while (vertex_hash_table[addr] != -1) addr = (addr + 1) % hash_table_size;
	vertex_hash_table[addr] = value;
}

int SearchHashTable(char *key)
{
	int addr = Hash(key);
	while (1)
	{
		if (vertex_hash_table[addr] == -1) return -1;
		if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
		addr = (addr + 1) % hash_table_size;
	}
	return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
	strcpy(vertex[num_vertices].name, name);
	vertex[num_vertices].degree = 0;
	num_vertices++;
	if (num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}
	InsertHashTable(name, num_vertices - 1);
	return num_vertices - 1;
}

/* Read network from the training file */
void ReadData()
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid;
	double weight;

	fin = fopen(network_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}
	num_edges = 0;
	while (fgets(str, sizeof(str), fin)) num_edges++;
	fclose(fin);
	printf("Number of edges: %lld          \n", num_edges);

	edge_source_id = (int *)malloc(num_edges*sizeof(int));
	edge_target_id = (int *)malloc(num_edges*sizeof(int));
	edge_weight = (double *)malloc(num_edges*sizeof(double));
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	fin = fopen(network_file, "rb");
	num_vertices = 0;
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		if (k % 10000 == 0)
		{
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}

		vid = SearchHashTable(name_v1);
		if (vid == -1) vid = AddVertex(name_v1);
		vertex[vid].degree += weight;
		edge_source_id[k] = vid;

		vid = SearchHashTable(name_v2);
		if (vid == -1) vid = AddVertex(name_v2);
		vertex[vid].degree += weight;
		edge_target_id[k] = vid;

		edge_weight[k] = weight;
	}
	fclose(fin);
	printf("Number of vertices: %lld          \n", num_vertices);
}

/* Read network from SAE graph */
void ReadDataFromSAE(MappedGraph *graph)
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid1,vid2;
	double weight;
    num_edges = graph->EdgeCount();
	num_vertices = 0;
	edge_source_id = (int *)malloc(num_edges*sizeof(int));
	edge_target_id = (int *)malloc(num_edges*sizeof(int));
	edge_weight = (double *)malloc(num_edges*sizeof(double));
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}
    int k = 0;
    for (auto itr = graph->Edges(); itr->Alive(); itr->Next()) {
		vid_t x = itr->Source()->GlobalId(), y = itr->Target()->GlobalId();
        
        strcpy(name_v1, sae::serialization::cvt_to_string(x).c_str());
        strcpy(name_v2, sae::serialization::cvt_to_string(y).c_str());
        weight = 1.0;
        vid1 = SearchHashTable(name_v1);
        if (vid1 == -1) vid1 = AddVertex(name_v1);
        vertex[vid1].degree += weight;
        edge_source_id[k] = vid1;

        vid2 = SearchHashTable(name_v2);
        if (vid2 == -1) vid2 = AddVertex(name_v2);
        vertex[vid2].degree += weight;
        edge_target_id[k] = vid2;
        //printf("%s:%d\t%s:%d\n",name_v1,vid1,name_v2,vid2);
        edge_weight[k++] = weight;
	}
}


/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable()
{
	alias = (long long *)malloc(num_edges*sizeof(long long));
	prob = (double *)malloc(num_edges*sizeof(double));
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
	for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

	for (long long k = num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block];
		alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

long long SampleAnEdge(double rand_value1, double rand_value2)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
	long long a, b;

	emb_vertex = (real *)malloc((long long)num_vertices * dim * sizeof(real));
	if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

	emb_context = (real *)malloc((long long)num_vertices * dim * sizeof(real));
	if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
		emb_context[a * dim + b] = 0;
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int));
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;
	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainLINEThread(void *id)
{
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	real *vec_error = (real *)calloc(dim, sizeof(real));

	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count>10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		curedge = SampleAnEdge(uni(), uni());
		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

		lu = u * dim;
		for (int c = 0; c != dim; c++) vec_error[c] = 0;

		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			if (d == 0)
			{
				target = v;
				label = 1;
			}
			else
			{
				target = neg_table[Rand(seed)];
				label = 0;
			}
			lv = target * dim;
			if (order == 1) Update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
			if (order == 2) Update(&emb_vertex[lu], &emb_context[lv], vec_error, label);
		}
		for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error[c];

		count++;
	}
	free(vec_error);
	return NULL;
}

void Output()
{
	FILE *fo = fopen(embedding_file, "wb");
	fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

void TrainLINE(MappedGraph* graph) {
	long a;
	boost::thread *pt = new boost::thread[num_threads];

	if (order != 1 && order != 2)
	{
		printf("Error: order should be eighther 1 or 2!\n");
		exit(1);
	}
	printf("--------------------------------\n");
	printf("Order: %d\n", order);
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("--------------------------------\n");

	InitHashTable();
	//ReadData();
    ReadDataFromSAE(graph);
	InitAliasTable();
	InitVector();
	InitNegTable();
	InitSigmoidTable();

	clock_t start = clock();
	printf("--------------------------------\n");
	for (a = 0; a < num_threads; a++) pt[a] = boost::thread(TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pt[a].join();
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	Output();
}

/* Add a vertex to the vertex set2 */
int AddVertex2(char *name, int vid)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[vid].name = (char *)calloc(length, sizeof(char));
	strcpy(vertex[vid].name, name);
	vertex[vid].degree = 0;
	InsertHashTable(name, vid);
	return vid;
}

void ReadVector()
{
	char ch, name[MAX_STRING];
	real f_num;
	long long l;

	FILE *fi = fopen(vector_file1, "rb");
	if (fi == NULL) {
		printf("Vector file 1 not found\n");
		exit(1);
	}
	fscanf(fi, "%lld %lld", &num_vertices, &vector_dim1);
	vertex = (struct ClassVertex *)calloc(num_vertices, sizeof(struct ClassVertex));
	vec1 = (real *)calloc(num_vertices * vector_dim1, sizeof(real));
	for (long long k = 0; k != num_vertices; k++)
	{
		fscanf(fi, "%s", name);
		ch = fgetc(fi);
		AddVertex2(name, k);
		l = k * vector_dim1;
		for (int c = 0; c != vector_dim1; c++)
		{
			//fread(&f_num, sizeof(real), 1, fi);
            fscanf(fi,"%lf",&f_num);
			vec1[c + l] = (real)f_num;
            //printf("vec1:%.5lf\t read:%.5lf\n",f_num,vec1[c+l]);
		}
	}
	fclose(fi);

	fi = fopen(vector_file2, "rb");
	if (fi == NULL) {
		printf("Vector file 2 not found\n");
		exit(1);
	}
	fscanf(fi, "%lld %lld", &l, &vector_dim2);
	vec2 = (real *)calloc((num_vertices + 1) * vector_dim2, sizeof(real));
	for (long long k = 0; k != num_vertices; k++)
	{
		fscanf(fi, "%s", name);
		ch = fgetc(fi);
		int i = SearchHashTable(name);
		if (i == -1) l = num_vertices * vector_dim2;
		else l = i * vector_dim2;
		for (int c = 0; c != vector_dim2; c++)
		{
			//fread(&f_num, sizeof(real), 1, fi);
            fscanf(fi,"%lf",&f_num);
			vec2[c + l] = (real)f_num;
		}
	}
	fclose(fi);

	printf("Vocab size: %lld\n", num_vertices);
	printf("Vector size 1: %lld\n", vector_dim1);
	printf("Vector size 2: %lld\n", vector_dim2);
}

std::vector<std::vector<double> > TrainModel() {
	long long a, b;
	double len;

	InitHashTable();
	ReadVector();
    std::vector<std::vector< double> > ret(num_vertices);
	for (a = 0; a < num_vertices; a++) {
		len = 0;
		for (b = 0; b < vector_dim1; b++) len += vec1[b + a * vector_dim1] * vec1[b + a * vector_dim1];
		len = sqrt(len);
		for (b = 0; b < vector_dim1; b++) vec1[b + a * vector_dim1] /= len;
        
		len = 0;
		for (b = 0; b < vector_dim2; b++) len += vec2[b + a * vector_dim2] * vec2[b + a * vector_dim2];
		len = sqrt(len);
		for (b = 0; b < vector_dim2; b++) vec2[b + a * vector_dim2] /= len;

        for (b = 0; b < vector_dim1; b++)
            ret[a].push_back(vec1[a * vector_dim1 + b]);
		for (b = 0; b < vector_dim2; b++)
            ret[a].push_back(vec2[a * vector_dim1 + b]);
	}
    return ret;
}


std::vector<std::vector<double> >  LINE::solve(int T, int d) {
    is_binary = 0;
    dim = d;
    order = 2;
    num_negative = 5;
    total_samples = T;
    init_rho = 0.025;
    num_threads = 20;
    total_samples *= 1000000;
	rho = init_rho;
	vertex = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	
    std::string out1 = "./output/vec1.txt";
    strcpy(embedding_file, out1.c_str());
    TrainLINE(graph);
    
    std::string out2 = "./output/vec2.txt";
    strcpy(embedding_file, out2.c_str());
    order = 1;
    TrainLINE(graph);
    
    strcpy(vector_file1,out1.c_str());
    strcpy(vector_file2,out2.c_str());
    
    std::vector<std::vector< double> > ans =  TrainModel();
    return ans;
}