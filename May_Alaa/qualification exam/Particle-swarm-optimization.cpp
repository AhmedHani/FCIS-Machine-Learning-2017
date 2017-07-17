#include <iostream>
#include<bits/stdc++.h>
using namespace std;
#define NUM_OF_Particles 10

// matrix class made it 5 * 5 by default for only the sample input in documentation
class matrix
{
public:
double arr[5][5];
int row;
int col;

matrix()
{
    row = 5 ; col = 5;


     for(int i = 0 ; i< 5 ; i++)
    {
        for(int y = 0 ; y< 5 ; y++)
        {

            this->arr[i][y] = 0;

        }
    }



}

matrix(int r,int c)
{
    row = r;
    col = c;
}


// overloading the equal operator to assign 2 matrices

void operator = (const matrix &M )
{
         for(int i = 0 ; i< M.row; i++)
         {

             for(int y = 0 ; y < M.col ; y++ )
             {

                 arr[i][y] = M.arr[i][y];

             }
         }
}


matrix operator + ( matrix &a)
{


    for(int i=0; i<a.row; i++)
    {
        for(int x=0; x<a.col; x++)
        {
            a.arr[i][x]+=this->arr[i][x];
        }

    }
return a ;

}



};



matrix Input_matrix;

// each particle will have a dimension of 5 x 2 because the dimension of the optimized matrices should be N * R
// N is the number of rows or columns in the square matrix , r should be less than (N*N)/(2N)
//according to this paper so we choose r = 2
//http://www.idrbt.ac.in/assets/alumni/PT-2011/BirendraKumar_ParticleSwarmOptimization_2011.pdf

class particle
{

 public:
     matrix position1; // 5 x 2
   matrix position2; // 2 x 5

   matrix best_pos1; // 5 x 2
   matrix best_pos2; // 2 x 5

   matrix velocity1; // 5 x 2
   matrix velocity2;  // 2 x 5


  particle() : position1(5,2),position2(2,5),best_pos1(5,2) ,best_pos2(2,5),velocity1(5,2),velocity2(2,5)
  {}


};

// declaring the best particle found which is the final answer
particle best_particle;

//declaring the array of particle (the entire swarm)

particle swarm[NUM_OF_Particles];

 // function for matrix multiplication
matrix multi_matrix(matrix X1,matrix X2)
{
    matrix result(X1.row,X2.col);

    int x=0;
    for(int z=0; z<X1.row; z++)
    {
        for(int c=0; c<X2.col; c++)
        {

            for(int p=0; p<X1.col; p++)
            {

                x+=X1.arr[z][p]*X2.arr[p][c];

            }
            result.arr[z][c]=x;

            x=0;

        }
    }
    return result;
}


// cost function is root-mean-square error between 2 matrices (RMSE)
 // N is the number of elements which is the dimension of the matrix (the two are of the same size)

double cost_function(matrix X1, matrix X2)
{

  // loop through the matrix
  double sum = 0 ;
  double N = X1.row * X1.col;
  double tmp;
    for(int i = 0 ; i < X1.row; i++)
    {

        for(int y = 0 ; y < X1.col; y++)
        {

            // get the square of the difference

            tmp = X1.arr[i][y] - X2.arr[i][y];
            tmp = tmp * tmp;

            sum += tmp;
        }

    }

 // get the mean square error

 double MSE = sum / N;
 // return root  mean square error
return sqrt(MSE);

}


// random generation function . takes the particle and assign the random values to position matrices or velocity matrices
void generate_random_and_assign(particle& P ,string choice,int dimension)
{

    //generate random elements using normal distribution

   vector<double> random_values;
   unsigned seed = time(0);
   std::default_random_engine rd(seed);
    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen(rd());

     // takes the mean and the standard deviation as arguments  , standard deviation is the square root of variance
     // mean = 0 , variance = 0.1 , standard deviation = sqrt(variance)
    normal_distribution<double>  random_generator (0.0,sqrt(0.1));
    double num;
    for(int i = 0 ; i< dimension ; i++)
    {
     num = random_generator(gen);

    random_values.push_back(num);
    }

   //assign random values according to the string

   if(choice == "position")
   {

       int c = 0;


       for(int i = 0 ; i< P.position1.row; i++)
       {
           for(int y = 0 ; y< P.position1.col; y++)
           {
               P.position1.arr[i][y] = random_values[c];
               c++;

           }
       }

         //traversing the first position matrix and assigning random values
       for(int i = 0 ; i< P.position2.row; i++)
       {
           for(int y = 0 ; y< P.position2.col; y++)
           {
               P.position2.arr[i][y] = random_values[c];
               c++;

           }
       }

   }
   else if (choice == "velocity")
   {
                  int c = 0;


       for(int i = 0 ; i< P.velocity1.row; i++)
       {
           for(int y = 0 ; y< P.velocity1.col; y++)
           {
               P.velocity1.arr[i][y] = random_values[c];
               c++;

           }
       }

         //traversing the first position matrix and assigning random values
       for(int i = 0 ; i< P.velocity2.row; i++)
       {
           for(int y = 0 ; y< P.velocity2.col; y++)
           {
               P.velocity2.arr[i][y] = random_values[c];
               c++;

           }
       }
   }
}


void update_velocity(particle& P)
{

    //looping on the first velocity matrix
    //updating the first velocity matrix
    for(int i = 0 ; i < P.velocity1.row; i++ )
    {
        for(int y = 0 ; y< P.velocity1.col; y++)
        {
            srand (static_cast <unsigned> (time(0)));

           //generating 2 pseudo random number between 0.0 and 1.0 inclusive
            float r1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            float r2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

            //according to the formula
        // vi,d ← ω vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)

           P.velocity1.arr[i][y] = P.velocity1.arr[i][y] + 2 * r1 * (P.best_pos1.arr[i][y] - P.position1.arr[i][y]) +
           2 * r2 * (best_particle.best_pos1.arr[i][y] - P.position1.arr[i][y] ) ;
        }

    }
    //the same goes for the second velocity matrix

    for(int i = 0 ; i < P.velocity2.row; i++ )
    {
        for(int y = 0 ; y< P.velocity2.col; y++)
        {
            srand (static_cast <unsigned> (time(0)));

           //generating 2 pseudo random number between 0.0 and 1.0 inclusive
            float r1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            float r2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

            //according to the formula
        // vi,d ← ω vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)

           P.velocity2.arr[i][y] = P.velocity2.arr[i][y] + 2 * r1 * (P.best_pos2.arr[i][y] - P.position2.arr[i][y]) +
           2 * r2 * (best_particle.best_pos2.arr[i][y] - P.position2.arr[i][y] ) ;
        }

    }

}



// optimization function

particle Optimize_Particles(int dimentions)
{

    // initializing the swarm

    // for  each particle
    //  1- generate  random numbers equal to the dimension ,assign values to position matrices
    //2- generate random numbers for equal to the dimension ,assign value to velocity matrices
    // 3- assign this initial position to the best position matrices
    //4- get the cost value(error) of best particle position using the cost function f(Pi)
    //5 get the cost value of the best particle position  in  the entire swarm f(G)
    //6- if f(G) > f(Pi) ---> G = Pi
    for( int i= 0; i < NUM_OF_Particles; i++)
    {




          // 1
          generate_random_and_assign(swarm[i],"position", dimentions);


          // 2
          generate_random_and_assign(swarm[i],"velocity", dimentions);



        //3
        swarm[i].best_pos1 = swarm[i].position1;
        swarm[i].best_pos2 = swarm[i].position2;



       //if it's the first time then initialize the swarm's best particle position with the first particle position in the swarm
        if(i == 0 )
        {
         best_particle.best_pos1 = swarm[i].best_pos1;
         best_particle.best_pos2 = swarm[i].best_pos2;

        }
        else
        {
            //compare the swarm's best particle position with the best particle position so far
            //4
            matrix Pi = multi_matrix(swarm[i].best_pos1,swarm[i].best_pos2);
            double f_Pi = cost_function(Pi,Input_matrix);
            //5
            matrix G = multi_matrix(best_particle.best_pos1,best_particle.best_pos2);
            double f_G = cost_function(G,Input_matrix);

            //6
            if(f_G > f_Pi)
            {
                best_particle.best_pos1 = swarm[i].best_pos1;
                best_particle.best_pos2 = swarm[i].best_pos2;
            }

        }

    }


    // looping --> each iteration we update the velocity matrix and get a new position for each particle

    // I set a maximum number of iteration (assuming that there should be a termination condition)

    int MAX_ITERATIONS = 100;

    int i = 1;

  while(i != MAX_ITERATIONS)
    {

    for( int i  = 0; i <= NUM_OF_Particles; i++)
    {

       //update particle's velocity according to the formula
       //vi,d = ω vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d) ---> w = 1 ,rp,rg are random numbers,φp and φg = 2



      update_velocity(swarm[i]);


      //update particle's position
      swarm[i].position1 =  swarm[i].position1 + swarm[i].velocity1;
      swarm[i].position2 =  swarm[i].position2 + swarm[i].velocity2;

      //if f(xi) < f(pi) then
        // Update the particle's best known position: pi = xi
         //if f(pi) < f(g) then
            //Update the swarm's best known position: g = pi

            matrix Pi = multi_matrix(swarm[i].best_pos1,swarm[i].best_pos2);
            double f_Pi = cost_function(Pi,Input_matrix);

            matrix xi = multi_matrix(swarm[i].position1,swarm[i].position2);
            double f_xi = cost_function(xi,Input_matrix);


            if(f_Pi > f_xi)
            {
                 swarm[i].best_pos1 = swarm[i].position1;
                 swarm[i].best_pos2 = swarm[i].best_pos1;

                  matrix G = multi_matrix(best_particle.best_pos1,best_particle.best_pos2);
                  double f_G = cost_function(G,Input_matrix);

                   Pi = multi_matrix(swarm[i].best_pos1,swarm[i].best_pos2);
                   f_Pi = cost_function(Pi,Input_matrix);

                    if (f_G > f_Pi)
                    {
                         best_particle.best_pos1 = swarm[i].best_pos1;
                         best_particle.best_pos2 = swarm[i].best_pos2;

                    }
            }

    }
    i++;

    }


return  best_particle; //final solution is in best_particle.best_pos1 & .best_pos2

}

int main()
{

  Input_matrix.row = 5;
  Input_matrix.col = 5;
cout<<"enter 5 X 5 matrix"<<'\n';

   for(int i = 0 ; i < 5 ; i++ )
   {
       for(int y = 0 ; y<5 ; y++)
       {

           cin>>Input_matrix.arr[i][y];
       }

   }

particle p = Optimize_Particles(10);

cout<<"final"<<'\n';
cout<<"first matrix size -->  "<<p.best_pos1.row<<" "<<p.best_pos1.col<<'\n';
 for(int i = 0 ; i < p.best_pos1.row ; i++ )
   {
       for(int y = 0 ; y<p.best_pos1.col ; y++)
       {

           cout<<p.best_pos1.arr[i][y]<<" ";
       }
       cout<<'\n';
   }
cout<<'\n';

  cout<<"second matrix size -->  "<<p.best_pos2.row<<" "<<p.best_pos2.col<<'\n';
   for(int i = 0 ; i < p.best_pos2.row ; i++ )
   {
       for(int y = 0 ; y<p.best_pos2.col ; y++)
       {

           cout<<p.best_pos2.arr[i][y]<<" ";
       }
       cout<<'\n';
   }




    return 0;
}
