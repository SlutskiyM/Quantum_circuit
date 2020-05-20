
#include <stdio.h>
#include <iostream>
#include <vector>
#include <random> 
#include <complex>
#include <math.h>
#include <fstream>
#include <utility>
#include <string>
#include <chrono>
#include <set>

#define X_GATE 1
#define Y_GATE 2
#define W_GATE 3
#define ISWAP  4

#define iterations 50

//#define PI 3.141592653589793
const double PI = std::acos(-1);

using namespace std;
const complex<double> imaginary(0.0, 1.0);

const complex<double> sq_X[4] = { 1 / sqrt(2), (1.0 - imaginary) / sqrt(2),
(1.0 - imaginary) / sqrt(2), 1 / sqrt(2) };

const complex<double> sq_Y[4] = { 1 / sqrt(2), (1.0 + imaginary) / sqrt(2),
1.0 / sqrt(2), 1 / sqrt(2) };

const complex<double> sq_W[4] = { 1 / sqrt(2), (1.0 + exp(imaginary*PI * (-0.75))) / sqrt(2),
(1.0 + exp(imaginary*PI * (-0.25))) / sqrt(2), 1 / sqrt(2) };

const complex<double> sq_SWAP[16] = { 1, 0, 0, 0,
0, 0, imaginary, 0,
0, imaginary, 0, 0,
0, 0, 0 ,1 };

const int X[4] = { 0,1,1,0 };
const int E[4] = { 1,0,0,1 };
const double d_E[4] = { 1,0,0,1 };


const int renumber[53] = { 23,14,22,32,8,13,21,31,40,3,7,12,20,30,39,
46,0,2,6,11,19,29,38,45,50,1,5,10,18,28,37,44,49,52,4,
9,17,27,36,43,48,51,16,26,35,42,47,15,25,34,41,24,33 };

const int renumber2[53] = { 16,25,17,9,37,26,18,10,4,35,27,19,11,5,1,47,
42,36,28,20,12,6,2,0,51,48,43,37,29,21,13,7,3,52,49,
44,38,30,22,14,8,50,45,39,31,23,15,46,40,32,24,41,33 };

const int first_layer[10] = { 1,3,5,6,9,9,8,6,4,2 };

const int second_layer[10] = { 1,3,5,7,9,9,8,5,4,2 };

template < typename T, typename K>
void mat_mult(vector<T> &result, vector<K> &scnd_mat);

template < typename T, typename G >
void tensor_mult(vector<T> &prev, const G *gate, int gate_num = 0);

template <typename T>
void print_sq_matrix(const vector<T> &mat);

void x_decomposition(const vector<complex<double>> &mat,
	vector<vector<complex<double>>> &d_coeff, vector<vector<int>> &p_coeff, bool ori = 0);

int split_bits(const long long int target, const int pos, const int step);

static std::random_device rd;
int seed = 1337;
static std::mt19937 rng(rd());

long long int dice(const long long int a, const long long int b)
{
	std::uniform_int_distribution<long long int> uid(a, b); // random dice
	return uid(rng); // use rng as a generator
}


void int_to_binary(int a, vector<bool> &str)
{
	str.clear();
	do
	{
		str.push_back(a % 2);
		a /= 2;
	} while (a != 0);
}

int binary_to_int(vector<bool> &str)
{
	int ans = 0;
	for (int i = str.size() - 1; i >= 0; i--)
		ans += str[i] * pow(2, i);
	return ans;
}


class layer
{
public:
	int gate_type;
	bool isHead;
	int cur_chain_length;

	int link[2]; // indexes of the two nearest qubits

	layer() {
		gate_type = 0;
		isHead = 0;
		cur_chain_length = 0;
		link[0] = -1;
		link[1] = -1;
	}

	void print_full()
	{
		cout << "gate_type = " << gate_type << "\t isHead = " << isHead
			<< "\t cur_chain_length = " << cur_chain_length << endl << endl;
		cout << "next_q =" << link[0] << "\t prev_q = " << link[1] << endl;
	}
};

class qubit
{
public:

	layer pattern[2];

	void print_full()
	{
		//pattern[0].print_full();
		pattern[1].print_full();
	}
};

class decomp
{
public:
	vector<vector<complex<double>>> D; // vector of diagonal elements (ordered)
	vector <vector<int>> P; // vector with places on which non-zero (ones) elements are, also ordered
};

class q_seq
{
public:
	vector<qubit> seq;
	vector<vector<int>> first_chains;
	vector<vector<int>> second_chains;

	vector<decomp> first_decomp;
	vector<decomp> second_decomp;

	q_seq()
	{
		qubit tmp;
		for (int i = 0; i < 53; i++)
			seq.push_back(tmp);
	}
	void gen_first_layer(const int start, const int num) {
		seq[start].pattern[0].isHead = 1;
		for (int i = start; i < start + num; i++)
		{
			seq[i].pattern[0].cur_chain_length = num;
			seq[i].pattern[0].link[0] = i + 1;
			seq[i].pattern[0].link[1] = i - 1;
		}
	}

	void gen_second_layer(const int start, const int num)
	{

		seq[renumber[start]].pattern[1].isHead = 1;
		for (int i = start; i < start + num; i++)
		{
			seq[renumber[i]].pattern[1].cur_chain_length = num;
			if (i + 1 < 53)
				seq[renumber[i]].pattern[1].link[0] = renumber[i + 1];
			if (i - 1 > -1)
				seq[renumber[i]].pattern[1].link[1] = renumber[i - 1];
		}
	}

	void gen_53()
	{
		gen_first_layer(0, 1);
		gen_first_layer(1, 3);
		gen_first_layer(4, 5);
		gen_first_layer(9, 6);
		gen_first_layer(15, 9);
		gen_first_layer(24, 9);
		gen_first_layer(33, 8);
		gen_first_layer(41, 6);
		gen_first_layer(47, 4);
		gen_first_layer(51, 2);
		seq[52].pattern[0].link[0] = -1;

		gen_second_layer(0, 1);
		gen_second_layer(1, 3);
		gen_second_layer(4, 5);
		gen_second_layer(9, 7);
		gen_second_layer(16, 9);
		gen_second_layer(25, 9);
		gen_second_layer(34, 8);
		gen_second_layer(42, 5);
		gen_second_layer(47, 4);
		gen_second_layer(51, 2);
		seq[23].pattern[1].link[1] = -1;
		seq[33].pattern[1].link[0] = -1;
	}

	void print_graph(const int layer)
	{
		int next = 0, start;
		if (!layer)
			start = 0;
		else
			start = renumber[0];
		ofstream fout("chains" + to_string(layer) + ".txt");
		while (next != -1)
		{
			fout << start;
			next = seq[start].pattern[layer].link[0];
			while (!seq[next].pattern[layer].isHead)
			{
				fout << "->" << next;
				next = seq[next].pattern[layer].link[0];
				if (next == -1)
					break;
			}
			fout << endl;
			start = next;
		}
		fout.close();
	}

	void gen_gates()
	{
		for (int i = 0; i < seq.size(); i++)
		{
			seq[i].pattern[0].gate_type = dice(1, 3);
			seq[i].pattern[1].gate_type = dice(1, 3);
			while (1)
			{
				seq[i].pattern[1].gate_type = dice(1, 3);
				if (seq[i].pattern[0].gate_type != seq[i].pattern[1].gate_type)
					break;
			}
		}
	}

	void fill_chains()
	{
		vector<int> tmp;
		//	first_chains.push_back(tmp);
		for (qubit i : seq)
		{
			if (i.pattern[0].isHead)
			{
				if (tmp.size() > 0)
					first_chains.push_back(tmp);
				tmp.clear();
			}
			tmp.push_back(i.pattern[0].gate_type);
		}
		first_chains.push_back(tmp);
		tmp.clear();

		for (int next : renumber)
		{
			if (seq[next].pattern[1].isHead)
			{
				if (tmp.size() > 0)
					second_chains.push_back(tmp);
				tmp.clear();
			}
			tmp.push_back(seq[next].pattern[1].gate_type);
		}
		second_chains.push_back(tmp);
	}

	void operator_decomposition()
	{
		// по очереди обрабатывать first chains
		// для каждой цепочки строить оператор
		// расскладывать оператор по базису
		// записывать в decomp
		first_decomp.resize(10);
		second_decomp.resize(10);
		vector < complex<double>> prev, fst_swp, snd_swp;


		for (int i = 0; i < first_chains.size(); i++)
		{
		//	cout << "i = " << i << endl;
			prev.push_back(1);
			for (int j = 0; j < first_chains[i].size(); j++)
			{
				auto tmp = first_chains[i][j];
				switch (first_chains[i][j])
				{
				case X_GATE:
					tensor_mult(prev, sq_X);
					break;
				case Y_GATE:
					tensor_mult(prev, sq_Y);
					break;
				case W_GATE:
					tensor_mult(prev, sq_W);
					break;
				default:
					cout << "ERROR: WRONG GATE TYPE" << endl;
				}
			}

			fst_swp.push_back(1);
			snd_swp.push_back(1);

			if (first_chains[i].size() % 2)
			{
				for (int j = 0; j + 1 < first_chains[i].size(); j += 2)
					tensor_mult(fst_swp, sq_SWAP, 1);
				tensor_mult(fst_swp, d_E);

				tensor_mult(snd_swp, d_E);
				for (int j = 1; j + 1 < first_chains[i].size(); j += 2)
					tensor_mult(snd_swp, sq_SWAP, 1);
			}
			else
			{
				for (int j = 0; j + 1 < first_chains[i].size(); j += 2)
					tensor_mult(fst_swp, sq_SWAP, 1);

				tensor_mult(snd_swp, d_E);
				for (int j = 1; j + 1 < first_chains[i].size(); j += 2)
					tensor_mult(snd_swp, sq_SWAP, 1);
				tensor_mult(snd_swp, d_E);
			}
			mat_mult(fst_swp, snd_swp);

			mat_mult(prev, fst_swp);
			x_decomposition(prev, first_decomp[i].D, first_decomp[i].P);
			prev.clear();
			fst_swp.clear();
			snd_swp.clear();
		}
		prev.clear(); fst_swp.clear(); snd_swp.clear();
		for (int i = 0; i < second_chains.size(); i++)
		{
		//	cout << "i = " << i << endl;
			prev.push_back(1);
			for (int j = 0; j < second_chains[i].size(); j++)
			{
				auto tmp = second_chains[i][j];
				switch (second_chains[i][j])
				{
				case X_GATE:
					tensor_mult(prev, sq_X);
					break;
				case Y_GATE:
					tensor_mult(prev, sq_Y);
					break;
				case W_GATE:
					tensor_mult(prev, sq_W);
					break;
				default:
					cout << "ERROR: WRONG GATE TYPE" << endl;
				}
			}

			fst_swp.push_back(1);
			snd_swp.push_back(1);
			if (i == 1)
				i = 1;
			if (second_chains[i].size() % 2)
			{
				for (int j = 0; j + 1 < second_chains[i].size(); j += 2)
					tensor_mult(fst_swp, sq_SWAP, 1);
				tensor_mult(fst_swp, d_E);

				tensor_mult(snd_swp, d_E);
				for (int j = 1; j + 1 < second_chains[i].size(); j += 2)
					tensor_mult(snd_swp, sq_SWAP, 1);
			}
			else
			{
				for (int j = 0; j + 1 < second_chains[i].size(); j += 2)
					tensor_mult(fst_swp, sq_SWAP, 1);
				tensor_mult(snd_swp, d_E);
				for (int j = 1; j + 1 < second_chains[i].size(); j += 2)
					tensor_mult(snd_swp, sq_SWAP, 1);
				tensor_mult(snd_swp, d_E);
			}
			mat_mult(fst_swp, snd_swp);
			mat_mult(prev, fst_swp);
			x_decomposition(prev, second_decomp[i].D, second_decomp[i].P);
			prev.clear();
			fst_swp.clear();
			snd_swp.clear();
		}
	}

	void find_state(const long long int fst_permut, const long long int snd_permut,
		vector<int> &ans_z0, vector<int> &ans_zfinal)
	{
		ans_zfinal.clear();
		ans_z0.clear();

		for (int i = 0; i < 53; i++)
		{
			ans_z0.push_back(split_bits(fst_permut, i, 1));
			//cout << split_bits(fst_permut, i, 1) << " ";
		}


		for (int i = 0; i < 53; i++)
		{
			ans_zfinal.push_back((split_bits(snd_permut, i, 1) + ans_z0[i]) % 2);
			//cout << split_bits(snd_permut, i, 1) << " ";
		}

	}


	complex<double> find_d(const long long int fst_permut, const long long int snd_permut, const vector<int> &state0, const vector<int> &state_final)
	{
		vector<int> rows, cols; // parse for each chain
		rows.push_back(split_bits(fst_permut, 0, 1));
		rows.push_back(split_bits(fst_permut, 1, 3));
		rows.push_back(split_bits(fst_permut, 4, 5));
		rows.push_back(split_bits(fst_permut, 9, 6));
		rows.push_back(split_bits(fst_permut, 15, 9));
		rows.push_back(split_bits(fst_permut, 24, 9));
		rows.push_back(split_bits(fst_permut, 33, 8));
		rows.push_back(split_bits(fst_permut, 41, 6));
		rows.push_back(split_bits(fst_permut, 47, 4));
		rows.push_back(split_bits(fst_permut, 51, 2));

		cols.push_back(split_bits(snd_permut, 0, 1));
		cols.push_back(split_bits(snd_permut, 1, 3));
		cols.push_back(split_bits(snd_permut, 4, 5));
		cols.push_back(split_bits(snd_permut, 9, 7));
		cols.push_back(split_bits(snd_permut, 16, 9));
		cols.push_back(split_bits(snd_permut, 25, 9));
		cols.push_back(split_bits(snd_permut, 34, 8));
		cols.push_back(split_bits(snd_permut, 42, 5));
		cols.push_back(split_bits(snd_permut, 47, 4));
		cols.push_back(split_bits(snd_permut, 51, 2));

		vector<bool> str;
		int pos;
		complex<double> ans = 1.0;

		for (int start = 0, i = 0, step; start < 53; i++, start += step) // first layer multiplication
		{
			step = first_layer[i];
			str.clear();
			for (int i = 0; i < step; i++)
				str.push_back(state0[start + i]);
			pos = binary_to_int(str);

			ans *= first_decomp[i].D[rows[i]][pos];
		}

		vector <int> tmp_st = state_final;
		for (int i = 0; i < 53; i++) // permutation
		{
			tmp_st[i] = state_final[renumber[i]];
		}

		for (int start = 0, i = 0, step; start < 53; i++, start += step) // second layer multiplication
		{
			step = second_layer[i];
			str.clear();
			for (int i = 0; i < step; i++)
				str.push_back(tmp_st[start + i]);
			pos = binary_to_int(str);

			ans *= second_decomp[i].D[cols[i]][pos];
		}
		return ans;
	}

	void form_decomp_file()
	{
		operator_decomposition();
		ofstream fout("fst_layer_" + to_string(seed) + ".txt");
		for (auto & i : first_decomp)
		{
			fout << i.D.size() << " " << i.D[0].size() << endl;
			for (auto &j : i.D)
			{
				for (auto &k : j)
					fout << k << " ";
				fout << endl;
			}
			fout << endl;
			fout << i.P.size() << " " << i.P[0].size() << endl;
			for (auto &j : i.P)
			{
				for (auto &k : j)
					fout << k << " ";
				fout << endl;
			}
			fout << endl;
		}
		fout.close();
		fout.open("snd_layer_" + to_string(seed) + ".txt");
		for (auto & i : second_decomp)
		{
			fout << i.D.size() << " " << i.D[0].size() << endl;
			for (auto &j : i.D)
			{
				for (auto &k : j)
					fout << k << " ";
				fout << endl;
			}
			fout << endl;
			fout << i.P.size() << " " << i.P[0].size() << endl;
			for (auto &j : i.P)
			{
				for (auto &k : j)
					fout << k << " ";
				fout << endl;
			}
			fout << endl;
		}
		fout.close();
	}

	void decomp_from_file()
	{
		first_decomp.clear();
		first_decomp.resize(10);
		second_decomp.clear();
		second_decomp.resize(10);
		ifstream in("fst_layer_" + to_string(seed) + ".txt");
		int mat_size = 0, num_of_mat = 0;
		for (int i = 0; i < 10; i++)
		{
			in >> num_of_mat >> mat_size;
			first_decomp[i].D.resize(num_of_mat);
			for (int j = 0; j < num_of_mat; j++)
			{
				first_decomp[i].D[j].resize(mat_size);
				for (int k = 0; k < mat_size; k++)
				{
					in >> first_decomp[i].D[j][k];
				}
			}
			in >> num_of_mat >> mat_size;
			first_decomp[i].P.resize(num_of_mat);
			for (int j = 0; j < num_of_mat; j++)
			{
				first_decomp[i].P[j].resize(mat_size);
				for (int k = 0; k < mat_size; k++)
				{
					in >> first_decomp[i].P[j][k];
				}
			}
		}
		in.close();
		in.open("snd_layer_" + to_string(seed) + ".txt");
		for (int i = 0; i < 10; i++)
		{
			in >> num_of_mat >> mat_size;
			second_decomp[i].D.resize(num_of_mat);
			for (int j = 0; j < num_of_mat; j++)
			{
				second_decomp[i].D[j].resize(mat_size);
				for (int k = 0; k < mat_size; k++)
				{
					in >> second_decomp[i].D[j][k];
				}
			}
			in >> num_of_mat >> mat_size;
			second_decomp[i].P.resize(num_of_mat);
			for (int j = 0; j < num_of_mat; j++)
			{
				second_decomp[i].P[j].resize(mat_size);
				for (int k = 0; k < mat_size; k++)
				{
					in >> second_decomp[i].P[j][k];
				}
			}
		}
		in.close();
	}
};

int split_bits(const long long int target, const int in_pos, const int step)
{
	long long int ans = 1;


	int pos = 53 - in_pos - step;
	for (int i = 0; i < step - 1; i++)
		ans = (1 | (ans << 1));
	for (int i = 0; i < pos; i++)
		ans = (ans << 1);
	ans = (ans & target);
	for (int i = 0; i < pos; i++)
		ans = (ans >> 1);
	return ans;
}

long long int test_bits(vector<int> &vec)
{
	long long int ans = 0;
	vector<bool> s;
	int_to_binary(ans, s);
	ans = (ans | vec[0]);
	ans = (ans << 3);
	int_to_binary(ans, s);
	ans = (ans | vec[1]);
	ans = (ans << 5);
	int_to_binary(ans, s);
	ans = (ans | vec[2]);
	ans = (ans << 6);
	int_to_binary(ans, s);
	ans = (ans | vec[3]);
	ans = (ans << 9);
	int_to_binary(ans, s);
	ans = (ans | vec[4]);
	ans = (ans << 9);
	int_to_binary(ans, s);
	ans = (ans | vec[5]);
	ans = (ans << 8);
	int_to_binary(ans, s);
	ans = (ans | vec[6]);
	ans = (ans << 6);
	int_to_binary(ans, s);
	ans = (ans | vec[7]);
	ans = (ans << 4);
	int_to_binary(ans, s);
	ans = (ans | vec[8]);
	ans = (ans << 2);
	int_to_binary(ans, s);
	ans = (ans | vec[9]);
	int_to_binary(ans, s);
	return ans;
}


template < typename T, typename G >
void tensor_mult(vector<T> &prev, const G *gate, int gate_num) // square matrixes only
{
	int gate_dim = 2;
	if (gate_num == 1)
		gate_dim = 4;

	vector<T> next;
	T test;
	int prev_dim = sqrt(prev.size());
	int result_dim = gate_dim * prev_dim;
	next.clear();
	next.assign(result_dim * result_dim, 0.0);

	for (int m = 0; m < prev_dim; m++)
		for (int k = 0; k < prev_dim; k++)
			for (int i = 0; i < gate_dim; i++)
				for (int j = 0; j < gate_dim; j++)
				{
					next[(k*gate_dim + j) + ((m* gate_dim + i)*result_dim)] =
						prev[k + m *prev_dim] * gate[j + i * gate_dim];
					//  complex*int = crash
				}
	prev.swap(next);
}

template < typename T, typename K>
void mat_mult(vector<T> &fst_mat, vector<K> &scnd_mat) // fst_mat = fst_mat * scnd_mat
{
	vector<T> result(fst_mat.size());
	int dim = sqrt(fst_mat.size());
	T sum = 0.0;
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
		{
			for (int k = 0; k < dim; k++)
				sum += fst_mat[k + i * dim] * scnd_mat[j + k*dim];
			result[j + i * dim] = sum;
			sum = 0.0;
		}
	result.swap(fst_mat);
}

void x_decomposition(const vector<complex<double>> &mat, vector<vector<complex<double>>> &d_coeff, vector<vector<int>> &p_coeff, bool ori)
{
	int dim = sqrt(mat.size());
	int dim1 = dim, k = 0;
	while (dim1 > 1)
	{
		dim1 = dim1 / 2;
		k++;
	}

	vector<bool> str(k);
	vector<int> p_mat, next, tm_p;

	vector<complex<double>>  tm_d;
	for (int i = 0; i < dim; i++)
	{
		p_mat.clear();
		p_mat.push_back(1);

		tm_d.clear();
		tm_p.clear();
		int_to_binary(i, str);

		while (str.size() < k)
			str.push_back(0);

		for (int j = 0; j < k; j++)
		{
			if (str[j])
				tensor_mult(p_mat, X);
			else
				tensor_mult(p_mat, E);
			//print_sq_matrix(p_mat);
		}

		for (int j = 0; j < p_mat.size(); j++)
		{
			if (p_mat[j] == 1)
			{
				tm_p.push_back(j);
				tm_d.push_back(mat[j]);
			}
		}
		d_coeff.push_back(tm_d);
		p_coeff.push_back(tm_p);
	}
}

template <typename T>
void print_sq_matrix(const vector<T> &mat)
{
	cout << endl << endl;
	int dim = sqrt(mat.size());

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			cout << mat[j + i * dim] << "\t";
		}
		cout << endl;
	}
	cout << endl << endl;
}

int break_n_fix(long long int &i, long long int &j,const int position = -1) // should be long
{
	int pos;
	if (position == -1)
		pos = dice(0, 52);
	else
		pos = position;

	long long int tmp = 1;
	tmp = (tmp << pos);
	if ((tmp & j) >> pos)
		j = (j& (~tmp));
	else
		j = (j | tmp);


	tmp = 1;
	tmp = (tmp << pos);
	if ((tmp & i) >> pos)
		i = (i& (~tmp));
	else
		i = (i | tmp);

	return pos;
}

double idea_test(q_seq &s, const long long int N) //    NEWEST
{
	chrono::steady_clock::time_point begin, end;

	long long int i, tmp_long;
	long long int j;
	int pos1, pos2, count = 0;
	vector<int> state0, state_final, tmp_vec;
	complex <double> tmp_complex1,tmp_complex2;
	set<pair<long long int, long long int>> my_set;
	double quadr_sum = 0.0, dij  = 0.0;

	begin = std::chrono::steady_clock::now();
//	cout << "----------------------------------------" << endl;
//	cout << "Start of the iteration" << endl;
	i = dice(0, pow(2, 53) - 1);
	j = dice(0, pow(2, 53) - 1);
	s.find_state(i, j, state0, state_final);
	
	for (long long int k = 0; k < N; k++)
	{
		for (int k = 0; k < 250; k++)
		{
			pos1 = break_n_fix(i, j);
			state0[52 - pos1] = 1 - state0[52 - pos1];
		}
		tmp_long = i;
		tmp_complex1 = (s.find_d(i, j, state0, state_final));
		quadr_sum += real(tmp_complex1*conj(tmp_complex1));

		for (int k = 0; k < 250; k++)
		{
			pos2 = break_n_fix(i, j);
			state0[52 - pos2] = 1 - state0[52 - pos2];
		}
//		if (my_set.insert(make_pair(tmp_long, i)).second == 0)
//		{
//			count++;
//		}

		tmp_complex2 = (s.find_d(i, j, state0, state_final));
		dij += real(tmp_complex1*conj(tmp_complex2));

	}
//	cout << "count = " << count << endl;
	end = std::chrono::steady_clock::now();
//	std::cout << "dij part takes  " <<
//		std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[sec]" << std::endl;

	double res;
	res = dij + quadr_sum / (pow(2, 106) - 1) ;

//	cout << "Answer is " << res << endl;
	return res;
}

int main(int argc, char *argv[])
{
//	cout << dice(1,5) << endl;
	q_seq s;
	s.gen_53();
	s.gen_gates();
	s.fill_chains();



	s.operator_decomposition();
	//s.decomp_from_file();
	//s.form_decomp_file();

//	cout << "Idea test:" << endl;
	int tst_count;
	chrono::steady_clock::time_point begin, end;
	for(long long int L = pow(10,atoi(argv[1])), k = 15; k < 16; L = L * 10, k++)
	{	
		tst_count = 0;
		begin = std::chrono::steady_clock::now();
		for (int i = 0; i < iterations; i++)
		{
			if (real(idea_test(s, L)) < 0)
				tst_count++;
		}
	cout <<  (double)tst_count/iterations << endl;
//	cout << L << endl;
	end = std::chrono::steady_clock::now();
	//std::cout << 
	//	std::chrono::duration_cast<std::chrono::minutes>(end - begin).count() << "[min]" << std::endl;	
	}
//	cout << endl << tst_count << " out of " << iterations << " are negative" << endl;
	
	return 0 ;
}


