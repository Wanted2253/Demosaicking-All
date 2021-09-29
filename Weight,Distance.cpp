#include <bits/stdc++.h>
using namespace std;
void distance_generator(vector<vector<double>> &distance,bool horizontalflag){
	int rowk,colk;
	int rowk2,colk2;
	int sum;
	for(int i=0;i<distance.size();i++){
		colk = i % 3;
		rowk = i / 3;
		sum = 0;
		for(int j=0;j<9;j++){
			colk2 = j % 3;
			rowk2 = j / 3;
			if(horizontalflag){
				if(colk2<=colk){
					sum = abs(colk2-colk) + abs(rowk2-rowk) + 3;
				}
				else{
					sum = 3 - abs(colk2-colk) + abs(rowk2-rowk);
				} 
			}
			else{
				sum = abs(colk2-colk) + abs(rowk2-rowk);
				if(rowk2<=rowk){
					sum = abs(colk2-colk) + abs(rowk2-rowk) + 3;
				}
				else{
					sum = 3 - abs(rowk2-rowk) + abs(colk2-colk);
				}
			}
			distance[i][j] = sum;
		}
		for(int j=9;j<18;j++){
			colk2 = (j-9) % 3;
			rowk2 = (j-9) / 3;
			if(horizontalflag){
				if(colk2>=colk){
					sum = abs(colk2-colk) + abs(rowk2-rowk) + 3;
				}
				else{
					sum = 3 - abs(colk2-colk) + abs(rowk2-rowk);
				} 
			}
			else{
				sum = abs(colk2-colk) + abs(rowk2-rowk);
				if(rowk2>=rowk){
					sum = abs(colk2-colk) + abs(rowk2-rowk) + 3;
				}
				else{
					sum = 3 - abs(rowk2-rowk) + abs(colk2-colk);
				}
			}
			distance[i][j] = sum;
		}
	}
}
void weight_generator(vector<vector<double>> &weights, vector<vector<double>> &distance, double k ) {
	double sum = 0;
	double factor = 1;
	for(int i=0;i<weights.size();i++){
		sum = 0;
		for(int j=0;j<weights[0].size();j++){
			sum += pow(distance[i][j],k);
		}
		factor = 1.00/sum;                                             
		for(int j=0;j<weights[0].size();j++){
			weights[i][j] = factor*pow(distance[i][j],k);
		}
	}
	return;
}
int main(){
    float k = 2;
    vector<vector<double>> weights_h( 9 , vector<double> (18, 0));
	vector<vector<double>> distance_h( 9 , vector<double> (18, 0));
	distance_generator(distance_h,1);
	weight_generator(weights_h,distance_h,k);
	for(int i=0;i<9;i++){
	    for(int j=0;j<18;j++){
	        cout<< distance_h[i][j] << " ";
	    }
	    cout << endl;
	}
	cout<< endl;
	for(int i=0;i<9;i++){
	    float sum = 0;
	    for(int j=0;j<18;j++){
	        sum += weights_h[i][j];
	        cout<< weights_h[i][j] << " ";
	    }
	    cout <<sum<<" "<< endl;
	}
}
3 2 1 4 3 2 5 4 3 3 4 5 4 5 6 5 6 7 
4 3 2 5 4 3 6 5 4 2 3 4 3 4 5 4 5 6 
5 4 3 6 5 4 7 6 5 1 2 3 2 3 4 3 4 5 
4 3 2 3 2 1 4 3 2 4 5 6 3 4 5 4 5 6 
5 4 3 4 3 2 5 4 3 3 4 5 2 3 4 3 4 5 
6 5 4 5 4 3 6 5 4 2 3 4 1 2 3 2 3 4 
5 4 3 4 3 2 3 2 1 5 6 7 4 5 6 3 4 5 
6 5 4 5 4 3 4 3 2 4 5 6 3 4 5 2 3 4 
7 6 5 6 5 4 5 4 3 3 4 5 2 3 4 1 2 3 

0.0272727 0.0121212 0.0030303 0.0484848 0.0272727 0.0121212 0.0757576 0.0484848 0.0272727 0.0272727 0.0484848 0.0757576 0.0484848 0.0757576 0.109091 0.0757576 0.109091 0.148485 1 
0.0512821 0.0288462 0.0128205 0.0801282 0.0512821 0.0288462 0.115385 0.0801282 0.0512821 0.0128205 0.0288462 0.0512821 0.0288462 0.0512821 0.0801282 0.0512821 0.0801282 0.115385 1 
0.0757576 0.0484848 0.0272727 0.109091 0.0757576 0.0484848 0.148485 0.109091 0.0757576 0.0030303 0.0121212 0.0272727 0.0121212 0.0272727 0.0484848 0.0272727 0.0484848 0.0757576 1 
0.057971 0.0326087 0.0144928 0.0326087 0.0144928 0.00362319 0.057971 0.0326087 0.0144928 0.057971 0.0905797 0.130435 0.0326087 0.057971 0.0905797 0.057971 0.0905797 0.130435 1 
0.0968992 0.0620155 0.0348837 0.0620155 0.0348837 0.0155039 0.0968992 0.0620155 0.0348837 0.0348837 0.0620155 0.0968992 0.0155039 0.0348837 0.0620155 0.0348837 0.0620155 0.0968992 1 
0.130435 0.0905797 0.057971 0.0905797 0.057971 0.0326087 0.130435 0.0905797 0.057971 0.0144928 0.0326087 0.057971 0.00362319 0.0144928 0.0326087 0.0144928 0.0326087 0.057971 1 
0.0757576 0.0484848 0.0272727 0.0484848 0.0272727 0.0121212 0.0272727 0.0121212 0.0030303 0.0757576 0.109091 0.148485 0.0484848 0.0757576 0.109091 0.0272727 0.0484848 0.0757576 1 
0.115385 0.0801282 0.0512821 0.0801282 0.0512821 0.0288462 0.0512821 0.0288462 0.0128205 0.0512821 0.0801282 0.115385 0.0288462 0.0512821 0.0801282 0.0128205 0.0288462 0.0512821 1 
0.148485 0.109091 0.0757576 0.109091 0.0757576 0.0484848 0.0757576 0.0484848 0.0272727 0.0272727 0.0484848 0.0757576 0.0121212 0.0272727 0.0484848 0.0030303 0.0121212 0.0272727 1 
