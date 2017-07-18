// the value of all mat this file should be log probability

#include <omp.h>

typedef float pfloat;

/*
this function is designed to calc the score of a specific route, used to calc the loss
route:	tag sequence
trans:	trans mat (ncondition*ncondition)
emit:	emit mat (seql*ncondition)
sos:	logp of trans from <sos> tag (ncondition)
eos:	logp of trans to <eos> tag (ncondition)
seql:	sequence length
ncondition:	number of states
return:	score of route
*/
pfloat oneRouteScore(int* route, pfloat** trans, pfloat** emit, pfloat* sos, pfloat* eos, int seql, int ncondition);
//batch version of oneRouteScore
void routeScore(int** route, pfloat** trans, pfloat*** emit, pfloat* sos, pfloat* eos, int bsize, int* seql, int ncondition, pfloat* rs);
/*
this function is designed to find the path with the highest score
trans:	trans mat (ncondition*ncondition)
emit:	emit mat (seql*ncondition)
sos:	logp of trans from <sos> tag (ncondition)
eos:	logp of trans to <eos> tag (ncondition)
seql:	sequence length
ncondition:	number of states
rcache:	cache for viterbi route ((seql-1)*ncondition), minus 1 because the transfer from last tag of the sequence to <eos> can be handled by maxind and maxvalue. rcache[step][tag] means that: the tag selected at step makes the tag choosed at step+1;
scache:	cache for probability (ncondition)
route:	the result route of highest score
return:	the score
*/pfloat oneViterbiRoute(pfloat** trans, pfloat** emit, pfloat* sos, pfloat* eos, int seql, int ncondition, int** rcache, pfloat* scache, int* route);
//batch version of oneViterbiRoute
void viterbiRoute(pfloat** trans, pfloat*** emit, pfloat* sos, pfloat* eos, int bsize, int* seql, int ncondition, int*** rcache, pfloat** scache, int** route, pfloat* score);
/*calc gradient of parameters
gold:	answer sequence
pred:	predict sequence
bsize:	batch size
seql:	sequence length
losses:	loss
grad:	grad to trans
gsos:	grad to <sos>
geos:	grad to <eos>
*/
void calcGrad(int** gold, int** pred, int bsize, int* seql, pfloat* losses, pfloat** grad, pfloat* gsos, pfloat* geos);
/*calc gradient of parameters
gscore:	score of answer sequence
pscore:	score of prediction sequence
bsize:	batch size
seql:	sequence length
avg:	average loss or not
losses:	loss
*/
pfloat getLoss(pfloat* gscore, pfloat* pscore, int bsize, int* seql, int avg, pfloat* losses);

pfloat oneRouteScore(int* route, pfloat** trans, pfloat** emit, pfloat* sos, pfloat* eos, int seql, int ncondition){
	pfloat rs = sos[route[0]] + emit[0][route[0]];
	int i;
	for (i = 1; i < seql, ++i){
		rs += trans[route[i-1]][route[i]] + emit[i][route[i]]
	}
	rs += eos[route[seql-1]];
	return rs;
}

void routeScore(int** route, pfloat** trans, pfloat*** emit, pfloat* sos, pfloat* eos, int bsize, int* seql, int ncondition, pfloat* rs){
	int i;
	#pragma omp parallel for
	for (i = 0; i < bsize; ++i){
		rs[i] = oneRouteScore(route[i], trans, emit[i], sos, eos, seql[i], ncondition);
	}
}

pfloat oneViterbiRoute(pfloat** trans, pfloat** emit, pfloat* sos, pfloat* eos, int seql, int ncondition, int** rcache, pfloat* scache, int* route){
	int i;
	for (i = 0; i < ncondition; ++i){
		scache[i] = sos[i];
	}
	for (i = 0; i < seql - 1; ++i){
		int j;
		for (j = 0; j < ncondition; ++j){
			int maxind = 0;
			pfloat maxvalue = scache[0] + trans[0][j];
			int k;
			for (k = 1; k < ncondition; ++k){
				pfloat _score = scache[k] + trans[k][j];
				if (_score > maxvalue){
					maxind = k;
					maxvalue = _score;
				}
			}
			rcache[i][j] = maxind;
			scache[j] = maxvalue + emit[i][j];
		}
	}
	int maxind = 0;
	pfloat maxvalue = scache[0] + eos[0] + emit[seql - 1][0];
	for (i = 1; i < ncondition; ++i){
		pfloat _score = scache[i] + eos[i] + emit[seql - 1][i];
		if (_score > maxvalue){
			maxind = i;
			maxvalue = _score;
		}
	}
	route[seql - 1] = maxind;
	for (i = seql - 2; i >= 0; --i){
		route[i] = rcache[i][route[i + 1]];
	}
	return maxvalue;
}

void viterbiRoute(pfloat** trans, pfloat*** emit, pfloat* sos, pfloat* eos, int bsize, int* seql, int ncondition, int*** rcache, pfloat** scache, int** route, pfloat* score){
	int i;
	#pragma omp parallel for
	for (i = 0; i < bsize; ++i){
		score[i] = oneViterbiRoute(trans, emit[i], sos, eos, seql[i], ncondition, rcache[i], scache[i], route[i]);
	}
}

void calcGrad(int** gold, int** pred, int bsize, int* seql, pfloat* losses, pfloat** grad, pfloat* gsos, pfloat* geos){
	int i;
	for (i = 0; i < bsize; ++i){
		pfloat loss = losses[i];
		if (gold[i][0] != pred[i][0]){
			gsos[pred[i][0]] += loss;
			gsos[gold[i][0]] -= loss;
		}
		int j;
		for (j = 1; j < seql[i]; ++j){
			grad[pred[i][j - 1]][pred[i][j]] += loss;
			grad[gold[i][j - 1]][gold[i][j]] -= loss;
		}
		if (gold[i][seql[i] - 1] != pred[i][seql[i] - 1]){
			geos[pred[i][seql[i] - 1]] += loss;
			geos[gold[i][seql[i] - 1]] -= loss;
		}
	}
}

pfloat getLoss(pfloat* gscore, pfloat* pscore, int bsize, int* seql, int avg, pfloat* losses){
	pfloat loss = 0;
	int i;
	for (i = 0; i < bsize; ++i){
		losses[i] = pscore[i] - gscore[i];
		loss += losses[i];
	}
	if (avg){
		int l = 0;
		for (i = 0; i < bsize; ++i){
			l += seql[i];
		}
		loss /= l;
	}
	return loss;
}
