#ifndef NNLIB_H
#define NNLIB_H


#include <bitset>
#include <ostream>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <fstream>

typedef double normfun ( double, double );



double valabs(double x){return x>0?x:-x;}

double sigmoidal ( double x, double a = 1 ) { return ( 1 / (1 + exp(-a*x))) ;}
double inv_sig ( double x, double a = 1 ) {return ( log(x/(1-x))/ a ); }


double hiptan ( double x, double a = 1 ) { return (2*sigmoidal(x,a) - 1);}
double Dsigmoidal ( double x, double a = 1 ) { double y = sigmoidal(x,a); return ( a*y*(1-y) );}
double Dhiptan ( double x, double a = 1 ) { return (2*Dsigmoidal(x,a));}
double id ( double x, double a = 1 ) { return ( x ) ;}
double one ( double x, double a = 1 ) { return ( 1 );}



typedef struct _DNormFun
{
	normfun  *fun;
	normfun *dfun;
}  DNormFun;

DNormFun StandFun[] = 
{
	{id, one},
	{sigmoidal,    Dsigmoidal},
	{hiptan,    Dhiptan},
};










/*
 * Two Layers Neural Networks, with fixed input-hidden-output units
 * The methods are
 *  0 - Linear
 *  1 - sigmoidal
 *  2 - hyperbolic tangent
 * */
class two_level_NN{
	public:
		
		two_level_NN (long in=1, long hi=1, long ou=1, long met=1, double par=1){
			srand(time(NULL));
			inputs = in; hidden = hi; outputs = ou; fun_type = met;
			f = StandFun[met].fun; df = StandFun[met].dfun; a = par;
			double Max_w = 1.4; // /hidden;
			assert(met<3);
			
			first_level_weights = new double* [hi] ;
			for(long i=0;i<hi;i++){
				first_level_weights[i] = new double [in];
				for(long j=0;j<in;j++)
					first_level_weights[i][j]=((double) (rand()%1000)/((double) 1000 / Max_w)) - (Max_w/2);
				}
				
			second_level_weights = new double* [ou] ;
			for(long i=0;i<ou;i++){
				second_level_weights[i] = new double [hi];
				for(long j=0;j<hi;j++)
					second_level_weights[i][j]=((double) (rand()%1000)/((double) 1000 / Max_w)) - (Max_w/2);
				}
			
			//  0->output 1->hidden 2->netoutput 3->nethidden 
			units_values = new double* [4];
			units_values[0] = new double [ou];
		    units_values[1] = new double [hi];
		    units_values[2] = new double [ou];
		    units_values[3] = new double [hi];
			
			};
		
		two_level_NN (char* path){
			FILE* fp;
			fp = fopen(path,"r");
			
			fscanf(fp, "%ld %ld %ld %ld %lf",&inputs,&hidden,&outputs,&fun_type,&a);
			f = StandFun[fun_type].fun; df = StandFun[fun_type].dfun; 

			first_level_weights = new double* [hidden] ;
			for (long i=0;i<hidden;i++){
				first_level_weights[i] = new double [inputs];
				for (long j=0;j<inputs;j++)
					fscanf(fp, "%lf", &first_level_weights[i][j]);
				}
				
			second_level_weights = new double* [outputs] ;
			for (long i=0;i<outputs;i++){
				second_level_weights[i] = new double [hidden];
				for (long j=0;j<hidden;j++)
					fscanf(fp, "%lf", &second_level_weights[i][j]);
				}	
			
			//  0->output 1->hidden 2->netoutput 3->nethidden 
			units_values = new double* [4];
			units_values[0] = new double [outputs];
		    units_values[1] = new double [hidden];
		    units_values[2] = new double [outputs];
		    units_values[3] = new double [hidden];
			
			fclose(fp);
			
			
			} 
		
		~two_level_NN(){
			for(long i=0;i<hidden;i++)
					delete[] first_level_weights[i];
			delete[] first_level_weights;	
			
			for(long i=0;i<outputs;i++)
					delete[] second_level_weights[i];
			delete[] second_level_weights;		
			
			for(long i=0;i<4;i++)
				delete[] units_values[i];
			delete[] units_values;
				
			};
	
		
		void print_NN(){
			
			
			std::cout << "\n\nInput Units: " << inputs << "\nHidden Units: " << hidden << "\nOutput Units: " << outputs << "\n\n";
			std::cout << "weights from input to hidden units:" << std::endl;
			for (long i=0;i<hidden;i++){
				std::cout << "to hidden unit " << i << ": ";
				for (long j=0;j<inputs;j++)
					std::cout << first_level_weights[i][j] << " ";
				std::cout << std::endl;
				}
			std::cout << std::endl << std::endl;
			
			std::cout << "weights from hidden to output units:" << std::endl;
			for (long i=0;i<outputs;i++){
				std::cout << "to output unit " << i << ": ";
				for (long j=0;j<hidden;j++)
					std::cout << second_level_weights[i][j] << " ";
				std::cout << std::endl;
				}
			std::cout << std::endl << std::endl;
			
			};
		
		void fprint_NN(char* path){
			
			FILE* fp;
			fp = fopen(path,"w");
			fprintf(fp, "Input Units: %ld\nHidden Units: %ld\nOutput Units: %ld\n\n",inputs,hidden,outputs);
			fprintf(fp, "weights from input to hidden units:\n");
			for (long i=0;i<hidden;i++){
				fprintf(fp, "to hidden unit %ld: ", i);
				for (long j=0;j<inputs;j++)
					fprintf(fp, "%f ", first_level_weights[i][j]);
				fprintf(fp, "\n");
				}
			fprintf(fp, "\n\n");
			
			fprintf(fp, "weights from hidden to output units:\n");
			for (long i=0;i<outputs;i++){
				fprintf(fp, "to output unit %ld: ", i);
				for (long j=0;j<hidden;j++)
					fprintf(fp, "%f ", second_level_weights[i][j]);
				fprintf(fp, "\n");
				}
			fprintf(fp, "\n\n");
			fclose(fp);
			};
		
		/*
		 * The syntax is 
		 * Input Hidden Output Met Par
		 * input -> hidden weights, one hidden unit a row
		 * hidden -> output weights, one output unit a row
		 * */
		void fprintf_NN(char* path){
			
			FILE* fp;
			fp = fopen(path,"w");
			fprintf(fp, "%ld %ld %ld %ld %f\n",inputs,hidden,outputs, fun_type, a);
			for (long i=0;i<hidden;i++){
				for (long j=0;j<inputs;j++)
					fprintf(fp, "%f ", first_level_weights[i][j]);
				fprintf(fp, "\n");
				}
			for (long i=0;i<outputs;i++){
				for (long j=0;j<hidden;j++)
					fprintf(fp, "%f ", second_level_weights[i][j]);
				fprintf(fp, "\n");
				}
			fclose(fp);				
			};
	
	
		/*
	
		void freadf_NN(char* path){
			
			
			FILE* fp;
			fp = fopen(path,"r");
			fscanf(fp, "%ld %ld %ld %ld %lf",&inputs,&hidden,&outputs,&fun_type,&a);
			for (long i=0;i<hidden;i++){
				for (long j=0;j<inputs;j++)
					fscanf(fp, "%lf", &first_level_weights[i][j]);
				}
			for (long i=0;i<outputs;i++){
				for (long j=0;j<hidden;j++)
					fscanf(fp, "%lf", &second_level_weights[i][j]);
				}	
			
			};
	
		*/
		
		
		
		void compute_NN ( double* in ) {
			
			long i,j;
			double sum = 0;
			
			for(i=0;i<hidden;i++){
				for(j=0;j<inputs;j++) sum += first_level_weights[i][j]*in[j];
				units_values[1][i] = f(sum,a);
				units_values[3][i] = sum;				
				sum = 0;
				}
			
			for(i=0;i<outputs;i++){
				for(j=0;j<hidden;j++) sum += second_level_weights[i][j]*units_values[1][j];
				units_values[0][i] = f(sum,a);
				units_values[2][i] = sum;					
				sum = 0;
				}
		
			
			};
		
		
		// n dati in input (Set) con le soluzioni (ans)
		// Sum of Squared Error
		double SSE (double** Set, double** ans, long n){
			
			double error = 0;
			long i,j;
			
			for(i=0; i<n; i++){
				compute_NN(Set[i]);
				for(j=0; j<outputs; j++)
					error += (units_values[0][j]-ans[i][j])*(units_values[0][j]-ans[i][j]);
				}
				
			
			return (error/2);
			
			};
	
		double SSE (double* Set, double* ans){
			
			double error = 0;
			long j;
			
			compute_NN(Set);
			for(j=0; j<outputs; j++)
				error += (units_values[0][j]-ans[j])*(units_values[0][j]-ans[j]);
	
			return (error/2);
			
			};
		
		
		// Mean Euclidean Error
		double MEE (double** Set, double** ans, long n){
			
			double error = 0;
			double point_error = 0;
			long i,j;
			
			for(i=0; i<n; i++){
				compute_NN(Set[i]);
				for(j=0; j<outputs; j++)
					point_error += (units_values[0][j]-ans[i][j])*(units_values[0][j]-ans[i][j]);
				point_error = sqrt(point_error);
				error += point_error;
				point_error = 0;
				}
				
			return (error/n);
			
			};
	
		
		double MEE (double* Set, double* ans){
			
			double error = 0;
			long j;
			
			compute_NN(Set);
			for(j=0; j<outputs; j++)
				error += (units_values[0][j]-ans[j])*(units_values[0][j]-ans[j]);
			error = sqrt(error);
			
			return (error);
			
			};
	
	
		
		// n dati in input (TR) con le soluzioni (ans)
		void batch_train (double learning, double** TR, double** ans, long n){
			
		
			
			long i,j,k,l;
			// i->input j->hidden k->output l->data
			
			double sum = 0;
			double sum2 = 0;
			double*** values = new double** [n];
			
			
			
			for(l=0;l<n;l++){
				
				compute_NN(TR[l]);				
				values[l] = new double* [4];
				values[l][0] = new double [outputs];
				values[l][1] = new double [hidden];
				values[l][2] = new double [outputs];
				values[l][3] = new double [hidden];
				for(k=0;k<outputs;k++){
						values[l][0][k] = units_values[0][k];  // O_lk 
						values[l][2][k] = units_values[2][k];  // net_lk
					}
				
				for(j=0;j<hidden;j++){
						values[l][1][j] = units_values[1][j];  // O_lj
						values[l][3][j] = units_values[3][j];  // net_lj
					}
				
				}
			
			
			
			double** delta = new double* [n];
			for(l=0;l<n;l++){ 
				delta[l] = new double [outputs];
				for(k=0;k<outputs;k++)
					delta[l][k] = (ans[l][k] - values[l][0][k])*df(values[l][2][k],a);
				}
				
			
			for(j=0;j<hidden;j++)
				for(i=0;i<inputs;i++){
					
					for(l=0;l<n;l++){
						
						for(k=0;k<outputs;k++)
							sum2 += delta[l][k]*second_level_weights[k][j];
						
						sum += TR[l][i]*df(values[l][3][j],a)*sum2;
						sum2 = 0;
						}
					
					first_level_weights[j][i] +=  learning*(sum - 0.01*first_level_weights[j][i]) ;
					sum = 0;	
					}
			
			for(k=0;k<outputs;k++)
				for(j=0;j<hidden;j++){
					
					for(l=0;l<n;l++)
						sum += delta[l][k]*values[l][1][j] ;
					
					second_level_weights[k][j] +=  learning*(sum - 0.01*second_level_weights[k][j]) ;
					sum = 0;
					}
			
			//std::cout <<  std::endl;
			for(l=0;l<n;l++){
				for(j=0;j<4;j++)
					delete[] values[l][j];
				delete[] values[l];
				delete[] delta[l];
				}
			delete[] values;
			delete[] delta;
			};
		
	
		void on_line_train (double learning, double** TR, double** ans, long n){
			long i,j;
			double** aux = new double* [1];
			aux[0] = new double [inputs];
			double** aux2 = new double* [1];
			aux2[0] = new double [inputs];
			for(i=0;i<n;i++){
				for(j=0;j<inputs;j++){
					aux[0][j] = TR[i][j]; aux2[0][j] = ans[i][j];
					}
				batch_train(learning, aux, aux2, 1);
				}
				
			
			delete[] aux[0];
			delete[] aux2[0];
			delete[] aux;
			delete[] aux2;
			
			}
	
		void train (double learning, double* TR, double* ans){
			double** aux = new double* [1];
			aux[0] = new double [inputs];
			double** aux2 = new double* [1];
			aux2[0] = new double [inputs];
			long i;
			
			for(i=0;i<inputs;i++){
				aux[0][i] = TR[i]; aux2[0][i] = ans[i];
				}
			
			batch_train(learning, aux, aux2, 1);
			
			delete[] aux[0];
			delete[] aux2[0];
			delete[] aux;
			delete[] aux2;
			
			
			};
	
		
		// ty=0 -> batch, ty=1 -> online
		void auto_train(long ty=0, double learning=0.01, double** TR=NULL, long n=0){
			
			two_level_NN aux(inputs,hidden,inputs,fun_type,a);
			long i,j;
			switch(ty){
				case 0:
					aux.batch_train(learning,TR,TR,n);
					break;
				case 1:
					aux.on_line_train(learning,TR,TR,n);
				}
			double** weights = aux.GetFirstLevelWeights();
			for(i=0;i<hidden;i++)
				for(j=0;j<inputs;j++)
					first_level_weights[i][j] = weights[i][j];
			
			}
		
		double** GetBothLevel(){return units_values;}
				
		double**  GetFirstLevelWeights(){return first_level_weights;}
		
		double**  GetSecondLevelWeights(){return second_level_weights;}
		
		
		void train_and_test(double** data, double** ans, long n, long test, long epochs, double learn){
			
			
			
			assert(test <= n);
			double** TS = new double* [test];
			double** ans_TS = new double* [test];
			double** TR = new double* [n-test];
			double** ans_TR = new double* [n-test];
			
			long i,j;
			for(i=0;i<test;i++){
				TS[i] = new double[inputs];
				for(j=0;j<inputs;j++) TS[i][j] = data[i][j];
				ans_TS[i] = new double[outputs];
				for(j=0;j<outputs;j++) ans_TS[i][j] = ans[i][j];
				}
			for(i=test;i<n;i++){
				TR[i-test] = new double[inputs];
				for(j=0;j<inputs;j++) TR[i-test][j] = data[i][j];
				ans_TR[i-test] = new double[outputs];
				for(j=0;j<outputs;j++) ans_TR[i-test][j] = ans[i][j];
				}
			
			
			double aux = 0;
			
			std::ofstream SSE_TR("SSE_TR.txt");
			std::ofstream SSE_TS("SSE_TS.txt");
			
			for(i=0;i<epochs && valabs(aux-SSE(TR,ans_TR,n-test)) > 0.00001;i++){
				
				aux = SSE(TR,ans_TR,n-test);
				batch_train(learn,TR,ans_TR,n-test);
				SSE_TR << SSE(TR,ans_TR,n-test)/(n-test) << std::endl;
				SSE_TS << SSE(TS,ans_TS,test)/test << std::endl;				
				
				}
			
			SSE_TR.close();
			SSE_TS.close();
			
			
			
			
			
			for(i=0;i<test;i++){
				delete[] TS[i];
				delete[] ans_TS[i];
				}
			delete[] TS;
			delete[] ans_TS;
			
			for(i=0;i<n-test;i++){
				delete[] TR[i];
				delete[] ans_TR[i];
				}
			delete[] TR;
			delete[] ans_TR;
			
			
			
			}
		
		
		
	protected: 
	
		double** second_level_weights;
		double** first_level_weights;
		long inputs;
		long hidden;
		long outputs;
		normfun* f;
		normfun* df;
		long fun_type;
		double a;
		double** units_values;



};



/*
 * 2 Layer NN with threshold units in input and output
 * */
class NN : public two_level_NN {
	
	public:
	
		NN(long in=1, long hi=1, long ou=1, long met=1, double par=1):
			two_level_NN(in+1,hi,ou+1,met,par) {};
			
		NN(char* path):
			two_level_NN(path) {};
		
		~NN(){};
		
		long* Get_Unit_Numbers(){
			long* sizes = new long[3];
			sizes[0]=inputs-1;
			sizes[1]=hidden;
			sizes[2]=outputs-1;
			return sizes;
			}
		
		double Get_Parameter(){
			return a;
			}
		
		
		void compute_NN ( double* in ) {
			
			double* true_in = new double[inputs];
			long i;
			true_in[inputs-1]=1;
			for(i=0;i<inputs-1;i++) true_in[i]=in[i];
			two_level_NN::compute_NN(true_in);
			delete[] true_in;
			
			};
			
		
		
		void print_NN(){
			
			
			std::cout << "\n\nInput Units: " << inputs << "\nHidden Units: " << hidden << "\nOutput Units: " << outputs << "\n\n";
			std::cout << "This NN has one input threshold unit and one output threshold unit\n\n";
			std::cout << "weights from input to hidden units:" << std::endl;
			for (long i=0;i<hidden;i++){
				std::cout << "to hidden unit " << i << ": ";
				for (long j=0;j<inputs;j++)
					std::cout << first_level_weights[i][j] << " ";
				std::cout << std::endl;
				}
			std::cout << std::endl << std::endl;
			
			std::cout << "weights from hidden to output units:" << std::endl;
			for (long i=0;i<outputs;i++){
				std::cout << "to output unit " << i << ": ";
				for (long j=0;j<hidden;j++)
					std::cout << second_level_weights[i][j] << " ";
				std::cout << std::endl;
				}
			std::cout << std::endl << std::endl;
			
			};
		
		
		// n dati in input (Set) con le soluzioni (ans)
		// Sum of Squared Errors
		double SSE (double** Set, double** ans, long n){
			
			double error = 0;
			long i,j;
			
			for(i=0; i<n; i++){
				compute_NN(Set[i]);
				for(j=0; j<outputs-1; j++)
					error += (units_values[0][j]-ans[i][j])*(units_values[0][j]-ans[i][j]);
				}
				
			
			return (error/2);
			
			};
	
		
		double SSE (double* Set, double* ans){
			
			double error = 0;
			long j;
			
			compute_NN(Set);
			for(j=0; j<outputs-1; j++)
				error += (units_values[0][j]-ans[j])*(units_values[0][j]-ans[j]);
	
			return (error/2);
			
			};
		
		
		// Mean Euclidean Error
		double MEE (double** Set, double** ans, long n){
			
			double error = 0;
			double point_error = 0;
			long i,j;
			
			for(i=0; i<n; i++){
				compute_NN(Set[i]);
				for(j=0; j<outputs-1; j++)
					point_error += (units_values[0][j]-ans[i][j])*(units_values[0][j]-ans[i][j]);
				point_error = sqrt(point_error);
				error += point_error;
				point_error = 0;
				}
				
			return (error/n);
			
			};
	
		
		double MEE (double* Set, double* ans){
			
			double error = 0;
			long j;
			
			compute_NN(Set);
			for(j=0; j<outputs-1; j++)
				error += (units_values[0][j]-ans[j])*(units_values[0][j]-ans[j]);
			error = sqrt(error);
			
			return (error);
			
			};
	
	
		
		
	
		// n dati in input (TR) con le soluzioni (ans)
		void batch_train (double learning, double** TR, double** ans, long n){
			
			long i,j;
			double** True_TR = new double*[n];
			double** True_ans = new double*[n];
			for(i=0;i<n;i++) {
				True_TR[i] = new double[inputs];
				True_ans[i] = new double[outputs];
				True_TR[i][inputs-1] = 1;
				True_ans[i][outputs-1] = 1;
				for(j=0;j<inputs-1;j++)
					True_TR[i][j] = TR[i][j];
				for(j=0;j<outputs-1;j++)
					True_ans[i][j] = ans[i][j];
				}
				
			two_level_NN::batch_train (learning, True_TR, True_ans, n); 
			
			
			for(i=0;i<n;i++) {
				delete[] True_TR[i];
				delete[] True_ans[i];				
				}
			delete[] True_TR;
			delete[] True_ans;
			
			
			}
	
		void on_line_train (double learning, double** TR, double** ans, long n){
				long i,j;
			double** True_TR = new double*[n];
			double** True_ans = new double*[n];
			for(i=0;i<n;i++) {
				True_TR[i] = new double[inputs];
				True_ans[i] = new double[outputs];
				True_TR[i][inputs-1] = 1;
				True_ans[i][outputs-1] = 1;
				for(j=0;j<inputs-1;j++)
					True_TR[i][j] = TR[i][j];
				for(j=0;j<outputs-1;j++)
					True_ans[i][j] = ans[i][j];
				}
				
			two_level_NN::on_line_train (learning, True_TR, True_ans, n); 
			
			
			for(i=0;i<n;i++) {
				delete[] True_TR[i];
				delete[] True_ans[i];				
				}
			delete[] True_TR;
			delete[] True_ans;			
			
			}
	
		void train (double learning, double* TR, double* ans){
			long i;
			double* true_TR = new double[inputs];
			double* true_ans = new double[outputs];
			true_TR[inputs-1]=1;
			true_ans[outputs-1]=1;
			for(i=0;i<inputs-1;i++) true_TR[i]=TR[i];
			for(i=0;i<outputs-1;i++) true_ans[i]=ans[i];
			two_level_NN::train(learning,true_TR,true_ans);
			delete[] true_TR;
			delete[] true_ans;
			}
			
			
		void auto_train(long ty=0, double learning=0.01, double** TR=NULL, long n=0){
			
			long i,j;
			double** True_TR = new double*[n];
			for(i=0;i<n;i++) {
				True_TR[i] = new double[inputs];
				True_TR[i][inputs-1] = 1;
				for(j=0;j<inputs-1;j++)
					True_TR[i][j] = TR[i][j];
				}
			two_level_NN::auto_train(ty,learning,True_TR,n);
			
			for(i=0;i<n;i++) 
				delete[] True_TR[i];
			delete[] True_TR;	
			
			}	
		
		
		
		
		void train_and_test(double** data, double** ans, long n, long test, long epochs, double learn){
			
			long i,j;
			double** True_data = new double*[n];
			double** True_ans = new double*[n];
			for(i=0;i<n;i++) {
				True_data[i] = new double[inputs];
				True_ans[i] = new double[outputs];
				True_data[i][inputs-1] = 1;
				True_ans[i][outputs-1] = 1;
				for(j=0;j<inputs-1;j++)
					True_data[i][j] = data[i][j];
				for(j=0;j<outputs-1;j++)
					True_ans[i][j] = ans[i][j];
				}
			
			
			two_level_NN::train_and_test(True_data, True_ans, n, test, epochs, learn);
			
			
			for(i=0;i<n;i++) {
				delete[] True_data[i];
				delete[] True_ans[i];				
				}
			delete[] True_data;
			delete[] True_ans;
			
			
			
			}
		
	
	
	};



	
#endif
