#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define WHITE 0
#define RED 1
#define BLUE 2
#define OUT 3
#define IN 4

// declera the functions
void creat_gird(int col, int row, int **grid_1D, int ***grid);
void init_grid(int n, int ***grid);
void print_grid(int n, int ***grid);
int* row_of_processor(int numprocs, int n, int t);
void red_move(int col, int row, int ***grid);
void blue_move(int col, int row, int ***grid);
int analyze_result(int ***grid, int *displs, int tile_number, int n, int t, float c, int numprocs, int myid);
void sequential_computation(int **grid_1D, int ***grid, int tile_number, int n, int t, float c, int max_iters, int numprocs, int *displs, int myid);
void self_check(int ***grid, int ***grid_copy, int n);

int main(int argc, char **argv)
{
  int *grid_1D;         		// one-dimension version of grid
  int **grid;           		// two-dimension grid
  int *grid_1D_copy;    	    // the copy of one-dimension version of grid
  int **grid_copy;     		    // the copy of the initial two-dimension grid
  int *sub_grid_1D;
  int **sub_grid;
  int n, t, max_iters;           // n-cell grid size, t-tile grid size, c-terminating threshold, max_iters- maximum number of iterations
  float c;
  int tile_number;      	    // the number of tile in the grid
  int myid;
  int numprocs;
  int i, j;

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  
  int finished_p = 0;     				    //the finished flag for processes
  int finished = 0;       			    	// the flag show whether the iteration finished or not
  int redcount = 0, bluecount = 0;          // the count of red and blue
  int n_iters = 0;                           // the iteration times

  if (argc != 5)
    {
      if (myid == 0)
        {
          printf("Illegal arguments.\n");
          printf("Please enter the command in the following format:\n");
          printf("mpirun -np [proc num] red_blue [cell grid size] [tile grid size] [terminating threshold] [maximum number of iterations]\n");
          printf("Note: [cell grid size] % [tile grid size] = 0; [proc num] <= [tile per side].\n");
        }
      goto EXIT;
    }

  n = atoi(argv[1]);
  t = atoi(argv[2]);
  c = atof(argv[3]);
  max_iters = atoi(argv[4]);

  if ((n % t != 0) || (numprocs > t))
    {
      if (myid == 0)
        {
          printf("Illegal arguments.\n");
          printf("Please enter the command in the following format:\n");
          printf("mpirun -np [proc num] red_blue [cell grid size] [tile grid size] [terminating threshold] [maximum number of iterations]\n");
          printf("Note: [cell grid size] % [tile grid size] = 0; [proc num] <= [tile per side].\n");
        }
      goto EXIT;
    }

  tile_number = t * t;
  int *row;
  row = row_of_processor(numprocs, n, t);
  int *recvcounts = malloc(sizeof(int) * numprocs);
  int *displs = malloc(sizeof(int) * numprocs);
  displs[0] = 0;
  int index = 0;
  for (i = 0; i < numprocs; i++)
    {
      recvcounts[i] = row[i] * n;
      index = index + recvcounts[i];
      displs[i] = index - recvcounts[i];
    }
  if (myid == 0)
    {
      creat_gird(n, n, &grid_1D, &grid);
      creat_gird(n, n, &grid_1D_copy, &grid_copy);
      init_grid(n, &grid);
      memcpy(grid_1D_copy, grid_1D, sizeof(int) * n * n);
	  printf("\n");
      printf("The initial grid: \n");
      print_grid(n, &grid);
    }
  tile_number = (t * t * row[myid]) / n;
  creat_gird(n, (row[myid] + 2), &sub_grid_1D, &sub_grid);

  MPI_Scatterv(grid_1D, recvcounts, displs, MPI_INT, &sub_grid_1D[n], recvcounts[myid], MPI_INT, 0, MPI_COMM_WORLD);

  while (!finished && n_iters < max_iters)
    {
      n_iters = n_iters + 1;
      MPI_Sendrecv(&sub_grid_1D[row[myid] * n], n, MPI_INT, (myid + 1) % numprocs, 1, &sub_grid_1D[0], n, MPI_INT, (myid - 1 + numprocs) % numprocs, 1, MPI_COMM_WORLD, &status);
      MPI_Sendrecv(&sub_grid_1D[n], n, MPI_INT, (myid - 1 + numprocs) % numprocs, 2, &sub_grid_1D[row[myid] * n + n], n, MPI_INT, (myid + 1) % numprocs, 2, MPI_COMM_WORLD, &status);
      red_move(n, row[myid] + 2, &sub_grid);
      blue_move(n, row[myid] + 2, &sub_grid);
      finished_p = analyze_result(&sub_grid, displs, tile_number, n, t, c, numprocs, myid);
      MPI_Allreduce(&finished_p, &finished, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

  MPI_Gatherv(&sub_grid_1D[n], recvcounts[myid], MPI_INT, grid_1D, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (myid == 0)
    {
      tile_number = t * t;
      printf("\n");
      printf("The parallel computation result: \n");
      printf("After %d interations, the final grid: \n", n_iters);
      print_grid(n, &grid);
      numprocs = 1;              //to do the movement sequential for self_check
      sequential_computation(&grid_1D_copy, &grid_copy, tile_number, n, t, c, max_iters, numprocs, displs, myid);
      self_check(&grid, &grid_copy, n);
    }
  EXIT:
  MPI_Finalize();
  return 0;
}

// create 2-D array and corresponding 1-D array
void creat_gird(int col, int row, int **grid_1D, int ***grid)
{
  int count = col * row;

  *grid_1D = (int*)malloc(sizeof(int) * count);
  *grid = (int**)malloc(sizeof(int *) * row);
  int i;

  for (i = 0; i < row; i++)
    {
      (*grid)[i] = &((*grid_1D)[i * col]);
    }
}

// initialize the grid
void init_grid(int n, int ***grid)
{
  time_t s;

  srand((unsigned)time(&s));
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        {
          (*grid)[i][j] = rand() % 3;
        }
    }
}

// print the grid
void print_grid(int n, int ***grid)
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        {
          printf("%d", (*grid)[i][j]);
        }
      printf("\n");
    }
}

// calculate the row distributed to each process
int* row_of_processor(int numprocs, int n, int t)
{
  int *row = (int*)malloc(sizeof(int) * numprocs);        // array 'row' store the number of row that each processor should take
  int base = t / numprocs;
  int remaining = t % numprocs;
  int i;

  // calculate total tiles that each processor should take
  for (i = 0; i < numprocs; i++)
    {
      row[i] = base;
    }
  for (i = 0; i < remaining; i++)
    {
      row[i] = row[i] + 1;
    }

  // calculate the rows that each processor should take
  for (i = 0; i < numprocs; i++)
    {
      row[i] = row[i] * (n / t);
    }

  return row;
}

// red color movement
void red_move(int col, int row, int ***grid)
{
  int i, j;

  for (i = 0; i < row; i++)
    {
      if ((*grid)[i][0] == 1 && (*grid)[i][1] == 0)
        {
          (*grid)[i][0] = 4;
          (*grid)[i][1] = 3;
        }
      for (j = 1; j < col; j++)
        {
          if ((*grid)[i][j] == 1 && ((*grid)[i][(j + 1) % col] == 0))
            {
              (*grid)[i][j] = 0;
              (*grid)[i][(j + 1) % col] = 3;
            }
          else if ((*grid)[i][j] == 3)
            (*grid)[i][j] = 1;
        }
      if ((*grid)[i][0] == 3)
        (*grid)[i][0] = 1;
      else if ((*grid)[i][0] == 4)
        (*grid)[i][0] = 0;
    }
}

// blue color movement
void blue_move(int col, int row, int ***grid)
{
  int i, j;

  for (j = 0; j < col; j++)
    {
      if ((*grid)[0][j] == 2 && (*grid)[1][j] == 0)
        {
          (*grid)[0][j] = 4;
          (*grid)[1][j] = 3;
        }
      for (i = 1; i < row; i++)
        {
          if ((*grid)[i][j] == 2 && (*grid)[(i + 1) % row][j] == 0)
            {
              (*grid)[i][j] = 0;
              (*grid)[(i + 1) % row][j] = 3;
            }
          else if ((*grid)[i][j] == 3)
            (*grid)[i][j] = 2;
        }
      if ((*grid)[0][j] == 3)
        (*grid)[0][j] = 2;
      else if ((*grid)[0][j] == 4)
        (*grid)[0][j] = 0;
    }
}

// check whether there is any tile over the threshold
int analyze_result(int ***grid, int *displs, int tile_number, int n, int t, float c, int numprocs, int myid)
{
  int tile_volume = (n * n) / (t * t);       
  int tile_row, tile_column;       
  int redcount = 0, bluecount = 0;
  float red_ratio, blue_ratio;
  int finished = 0;
  int i, j, k;

//the method of parallel and sequential is different
  if (numprocs == 1)
    {
      tile_number = t * t;
      for (i = 0; i < tile_number; i++)
        {
          tile_row = i / t;
          tile_column = i % t;
          for (j = (n / t) * tile_row; j < (n / t) * tile_row + (n / t); j++)
            {
              for (k = (n / t) * tile_column; k < (n / t) * tile_column + (n / t); k++)
                {
                    
                  if ((*grid)[j][k] == 1)
                    {
                      redcount = redcount + 1;
                    }
                  if ((*grid)[j][k] == 2)
                    {
                      bluecount = bluecount + 1;
                    }                  
                }
            }
          red_ratio = redcount * 100 / tile_volume;
          blue_ratio = bluecount * 100 / tile_volume;

          if (blue_ratio > c && red_ratio > c)
            {
              printf("In tile %d, both the red and the blue color over the threshold.\n", i + 1);
              finished = 1;
            }

          if (red_ratio > c && blue_ratio < c)
            {
              printf("In tile %d, the red color over the threshold.\n", i + 1);
              finished = 1;
            }

          if (blue_ratio > c && red_ratio < c)
            {
              printf("In tile %d, the blue color over the threshold.\n", i + 1);
              finished = 1;
            }
          redcount = 0;
          bluecount = 0;
        }
    }
  else
    {
      for (i = 0; i < tile_number; i++)
        {
          tile_row = i / t;
          tile_column = i % t;
          for (j = (n / t) * tile_row; j < (n / t) * tile_row + (n / t); j++)
            {
              for (k = (n / t) * tile_column; k < (n / t) * tile_column + (n / t); k++)
                {
                  {
                    if ((*grid)[j + 1][k] == 1)
                      {
                        redcount = redcount + 1;
                      }
                    if ((*grid)[j + 1][k] == 2)
                      {
                        bluecount = bluecount + 1;
                      }
                  }
                }
            }

          red_ratio = redcount * 100 / tile_volume;
          blue_ratio = bluecount * 100 / tile_volume;
          int index;
          index = displs[myid] / (n * n / t / t);
          if (blue_ratio > c && red_ratio > c)
            {
              printf("In tile %d, both the red and the blue color over the threshold.\n", i + 1);
              finished = 1;
            }

          if (red_ratio > c && blue_ratio < c)
            {
              printf("In tile %d, the red color over the threshold.\n", i + 1);
              finished = 1;
            }

          if (blue_ratio > c && red_ratio < c)
            {
              printf("In tile %d, the blue color over the threshold.\n", i + 1);
              finished = 1;
            }
          redcount = 0;
          bluecount = 0;
		}
    }
  return finished;
}

// sequential movement
void sequential_computation(int **grid_1D, int ***grid, int tile_number, int n, int t, float c, int max_iters, int numprocs, int *displs, int myid)
{
  int finished = 0;
  int n_iters = 0;
  while (!finished && n_iters < max_iters)
    {
      n_iters = n_iters + 1;// count for the iteration number

      red_move(n, n, grid);
      blue_move(n, n, grid);

      finished = analyze_result(grid, displs, tile_number, n, t, c, 1, myid);
    }

  printf("\n");
  printf("The sequential computation result: \n");
  print_grid(n, grid);
}

// self-checking program
void self_check(int ***grid, int ***grid_copy, int n)
{
  int flag = 0;
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        {
          if ((*grid)[i][j] != (*grid_copy)[i][j])
            {
              flag = 1;
              break;
            }
          if (flag == 1)
            {
              break;
            }
        }
    }

  printf("\n");
  if (flag == 0)
    {
      printf("Self-chech: The result of parallel program is the same as sequential program.\n");
	  printf("            Means the result of the parallel movement is correct.\n");
    }
  else
    {
      printf("Self-chech: The result of parallel program is not the same as sequential program.\n");
	  printf("            Means the result of the parallel movement is wrong.\n");
    }
  printf("\n");
}