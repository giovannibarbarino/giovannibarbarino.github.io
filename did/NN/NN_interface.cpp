#include <bitset>
#include <ostream>
#include <iostream>
#include <vector>
#include <stdint.h>
#include "nnlib.h"
#include <math.h>
#include <string>
#include <sstream>

using namespace std;



NN* NN_loader(){
	string aux;
	cout << "\nEnter the path of your Neural Network: "; 
	cin >> aux;
	char* path = (char*)aux.c_str();
	NN* Nene = new NN(path);
	cout << "\nThe Neural Network has been loaded.\n";
	return Nene;
	}

NN* NN_creator(){
	
	long in, hi, ou, met;
	double par;
	
	cout << "\nPlease, enter the following informations\nInput Units: ";
	cin >> in;
	cout << "Hidden Units: ";
	cin >> hi;
	cout << "Output Units: ";
	cin >> ou;
	cout << "Type of truncation function (0-Linear, 1-Sigmoidal, 2-htan), Standard is 1: ";
	cin >> met;
	cout << "Parameter of the function, Standard is 1 : ";
	cin >> par;
	
	NN* Nene = new NN(in,hi,ou,met,par);
	return Nene;
	}

NN* introduction(){
	
	//NN* Nene;
	int a=0;
	//char* path = NULL;
	
	
	while(!(a==2 || a==1)){
	
	cout << "\nWhat do you want to do?\n1 - Create a new Neural Network\n2 - Load an existing Neural Network\nChoose (1/2): ";
	cin >> a;
	
	if(a==1){
		return NN_creator();
		}
	else if(a==2){
		return NN_loader();
		//cout << "\nEnter the path of your Neural Network: "; 
		//cin >> path;
		//Nene = new NN(path);
		//return Nene;
		}
	else {
		cout << "\nPlease, type 1 or 2\n\n";
		//return introduction();
		}
	}
	
	return NULL;
	
	}

int exiting(){
	int a=-1;
	
	cout << "\nDo you really want to go? Have you Saved? (1-Yes, 0-No): ";
	cin >> a;
	
	if(a) {
		a=7;
		cout << "\nGoodbye!";
		}
	
	return a;
	}

int throwing(){
	int a=-1;
	
	cout << "\nHave you Saved this Neural Network? (1-Yes, 0-No): ";
	cin >> a;
	
	if(a) 
		a=6;
	
	return a;
	}

void save_NN(NN* NeNe){
	string aux;
	cout << "\nPlease, specify the path where you want to save this NN: ";
	cin >> aux;
	char* path = (char*)aux.c_str();
	NeNe->fprintf_NN(path);
	cout << "\nThe Neural Network has been saved.\n";
	
	}

void comp_NN(NN* NeNe){
	long* sizes = NeNe->Get_Unit_Numbers();
	long in = sizes[0];
	long ou = sizes[2];
	long i, n;
	double* data = new double[in];
	cout << "\nPlease, type the " << in << " inputs\n"; 
	for(i=0;i<in;i++){
		cout << i+1 << ": ";
		cin >> data[i];
	}
	double** ress;
	NeNe->compute_NN(data);
	ress = NeNe->GetBothLevel();
	cout << "\nDo you want to denormalize the output? (1-yes, 0-no): ";
	cin >> n;
	
	
	for(i=0;i<ou;i++){
		if(n)
			cout << inv_sig(ress[0][i]) << "\t";
		else
			cout << ress[0][i] << "\t";
	}
	cout << endl;
	}


/*
 * The input file must be in the format
 * # inputs
 * rows of inputs 
*/
void mult_comp_NN(NN* NeNe){
	string aux;
	
	cout << "\nPlease, insert a file text with the inputs: ";
	cin >> aux;
	char* pathin = (char*)aux.c_str();
	FILE* fpi;
	fpi = fopen(pathin, "r");
	
	cout << "\nWhere do you want to save the outputs? ";
	cin >> aux;
	char* pathout = (char*)aux.c_str();
	FILE* fpo;
	fpo = fopen(pathout, "w");
	
	long n, m, i,j;
	fscanf(fpi,"%ld",&n);
	fprintf(fpo,"%ld\n",n);
	
	long* sizes = NeNe->Get_Unit_Numbers();
	long in = sizes[0];
	long out = sizes[2];
	double* data = new double[in];
	double** ress;
	
	cout << "\nDo you want to denormalize the output? (1-yes, 0-no): ";
	cin >> m;
	
	for(j=0;j<n;j++){
	
		for(i=0;i<in;i++)
			fscanf(fpi,"%lf",&data[i]);
		NeNe->compute_NN(data);
		ress = NeNe->GetBothLevel();
		for(i=0;i<out;i++){
			if(m)
				fprintf(fpo,"%lf\t",inv_sig(ress[0][i]));
			else
				fprintf(fpo,"%lf\t",ress[0][i]);
			}
		fprintf(fpo,"\n");

	}

	fclose(fpi);
	fclose(fpo);
	
	
	
	}

void Manual_Train(NN* NeNe,int n){
	
	long* sizes = NeNe->Get_Unit_Numbers();
	long in = sizes[0];
	long ou = sizes[2];
	long i;
	double* datain = new double[in];
	double* dataou = new double[ou];
	
	cout << "\nPlease, write the " << in << " input data:\n";
	for(i=0;i<in;i++){
		cout << "Input "<< i+1 <<": ";
		cin >> datain[i];
		}
	cout << "\nPlease, write the " << ou << " output data:\n";
	for(i=0;i<ou;i++){
		cout << "Output "<< i+1 <<": ";
		cin >> dataou[i];
		}	
	if(n)
		for(i=0;i<ou;i++)
			dataou[i] = sigmoidal(dataou[i],NeNe->Get_Parameter());		
	double lea;
	cout << "\nPlease, insert the learning rate (usually 0.01): ";
	cin >> lea;
	long epo;
	cout << "\nPlease, insert the number of epo (Standard is 1): ";
	cin >> epo;
	
	for (i=0;i<epo;i++)
		NeNe->train(lea,datain,dataou);
	
	cout << "\nTrain Completed!\n";
	}


/*
 * The input file must be in the format
 * # train data
 * all rows of inputs
 * all rows of outputs 
*/
void File_Train(NN* NeNe, int n, int m){
	
	string aux;
	
	cout << "\nPlease, insert a file text with the Training Set: ";
	cin >> aux;
	char* pathin = (char*)aux.c_str();
	FILE* fpi;
	fpi = fopen(pathin, "r");
	
	long* sizes = NeNe->Get_Unit_Numbers();
	long in = sizes[0];
	long ou = sizes[2];
	float a = NeNe->Get_Parameter();
	
	long num, i,j;
	fscanf(fpi,"%ld",&num);
	
	double** datain = new double* [num];
	double** dataou = new double* [num];
	for(i=0;i<num;i++){
		datain[i] = new double [in];
		dataou[i] = new double [ou];
	}
	
	for(j=0;j<num;j++)
		for(i=0;i<in;i++)
			fscanf(fpi,"%lf",&datain[j][i]);
	for(j=0;j<num;j++)
		for(i=0;i<ou;i++){
			fscanf(fpi,"%lf",&dataou[j][i]);
			if(n)
				dataou[j][i] = sigmoidal(dataou[j][i],a);
			}
	
	double lea;
	cout << "\nPlease, insert the learning rate (usually 0.01): ";
	cin >> lea;
	long epo;
	cout << "\nPlease, insert the number of epo (Standard is 1): ";
	cin >> epo;
	
	
	switch(m){
				case 1:
					for(i=0;i<epo;i++)
						NeNe->batch_train(lea,datain,dataou,num);
					break;
				case 0:
					for(i=0;i<epo;i++)
						NeNe->on_line_train(lea,datain,dataou,num);
				}


	fclose(fpi);
	
	}


int choice(NN* NeNe){
	
	int a=0;
	int n,m;
	
	while(a!=7){
		
		
		
	cout << "\nWhat do you want to do with this Neural Network?\n0 - Print it!\n1 - Train Manually\n2 - Train from File\n3 - Output on single Input\n4 - Save on File the Outputs on several Inputs from File\n5 - Save the actual Neural Network\n6 - Start a new Neural Network\n7 - Exit\nYour Choice(0-7): ";
	cin >> a;
	
	
	
	if(a==0){
		NeNe->print_NN();
		continue;
		}
	else if(a==7){
		a = exiting();
		continue;
	}
	else if(a==6){
		a = throwing();
		if(a){
			a = 6;
			break;
		}
		continue;
		}
	else if(a==5){
		save_NN(NeNe);
		continue;
		}
	else if(a==4){
		mult_comp_NN(NeNe);	
		continue;	
		}
	else if(a==3){
		comp_NN(NeNe);
		continue;
		}
	else if(a==2){
		cout << "\nDo you want to normalize your output? (1-yes, 0-no): ";
		cin >> n;
		cout << "\nDo you want a Batch or Online Training? (1-Batch, 0-Online): ";
		cin >> m;
		File_Train(NeNe,n,m);
		continue;
		}
	else if(a==1){
		cout << "\nDo you want to normalize your output? (1-yes, 0-no): ";
		cin >> n;
		Manual_Train(NeNe,n);
		continue;
		}
	else{
		cout << "Please, type a number in the interval 0-7";
		continue;
		}
	
		
		
	}
	
	
	
	return a;
	}


int main ()
{
	int a=0;
	NN* NeNe = NULL;
	
	cout << "\nWelcome to the terminal Neural Network Interface.\n\n";
	
	
	while(a!=7){
	
	delete NeNe;
	NN* NeNe = NULL;
	NeNe = introduction();
	a = choice(NeNe);
	
	}
	
	
	
	
	
	
	
	
	
	
	
	return 0;

}
