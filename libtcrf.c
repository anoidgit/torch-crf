// the value of all mat this file should be log probability

#include <omp.h>

/*
this function is disigned to calc the score of a specific route, used to calc the loss
route:	tag sequence
trans:	trans mat (ncondition*ncondition)
emit:	emit mat (seql*ncondition)
start:	logp of trans from <start> tag (ncondition)
end:	logp of trans to <end> tag (ncondition)
seql:	sequence length
ncondition:	number of states
return:	score of route
*/
float oneRouteScore(int* route, float** trans, float** emit, float* start, float* end, int seql, int ncondition);
//batch version of oneRouteScore
void routeScore(int** route, float** trans, float*** emit, float* start, float* end, int* seql, int ncondition, float* rs);
/*
this function is disigned to find the path with the highest score
trans:	trans mat (ncondition*ncondition)
emit:	emit mat (seql*ncondition)
start:	logp of trans from <start> tag (ncondition)
end:	logp of trans to <end> tag (ncondition)
seql:	sequence length
ncondition:	number of states
route:	the result route of highest score
return:	the score
*/float oneViterbiRoute(float** trans, float** emit, float* start, float* end, int seql, int ncondition, int** rcache, float* scache, int* route);
//batch version of oneViterbiRoute
void viterbiRoute(float** trans, float*** emit, float* start, float* end, int* seql, int ncondition, int*** rcache, float** scache, int** route, float* score);

float oneRouteScore(int* route, float** trans, float** emit, float* start, float* end, int seql, int ncondition){
	float rs = start[route[0]] + emit[0][route[0]];
	int i = 0;
	for (i = 1; i < seql, ++i){
		rs += trans[route[i-1]][route[i]] + emit[i][route[i]]
	}
	rs += end[route[seql-1]];
	return rs;
}

void routeScore(int** route, float** trans, float*** emit, float* start, float* end, int* seql, int ncondition, float* rs){
	int i = 0;
	#pragma omp parallel for
	for(i = 0; i < seql; ++i){
		rs[i] = oneRouteScore(route[i], trans, emit[i], start, end, seql[i], ncondition);
	}
}

// to be implemented
float oneViterbiRoute(float** trans, float** emit, float* start, float* end, int seql, int ncondition, int** rcache, float* scache, int* route){
	return 0;
}

void viterbiRoute(float** trans, float*** emit, float* start, float* end, int* seql, int ncondition, int*** rcache, float** scache, int** route, float* score){
	int i = 0;
	#pragma omp parallel for
	for(i = 0; i < seql; ++i){
		score[i] = oneViterbiRoute(trans, emit[i], start, end, seql[i], ncondition, rcache[i], scache[i], route[i]);
	}
}
