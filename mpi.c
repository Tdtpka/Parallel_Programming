#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX 300
#define EPS 1e-12

// Mỗi tiến trình lưu tối đa MAX hàng
double local[MAX][2 * MAX];
double pivotrow[2 * MAX];

int main(int argc, char **argv)
{
    int N, rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Tiến trình 0 nhập ma trận
    double A[MAX][MAX];
    if (rank == 0)
    {
        FILE *f = fopen("mpi.txt", "r");
        fscanf(f, "%d", &N);
        int choice;
        fscanf(f, "%d", &choice);
        if (choice == 1)
        {
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    fscanf(f, "%lf", &A[i][j]);
        }
        else
        {
            srand(0);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    A[i][j] = (i == j) ? 100.0 : (rand() % 3 + 1); // chéo lớn
        }
        fclose(f);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N <= 0 || N > MAX)
    {
        if (rank == 0)
            printf("N khong hop le!\n");
        MPI_Finalize();
        return 0;
    }

    int local_rows = 0;
    int row_index[MAX]; // lưu chỉ số hàng toàn cục mà tiến trình này giữ

    // Phân phối hàng: i % size == rank
    for (int i = 0; i < N; i++)
    {
        if (i % size == rank)
        {
            row_index[local_rows++] = i;
        }
    }

    // Gửi từng hàng đến tiến trình tương ứng
    for (int i = 0; i < N; i++)
    {
        int dest = i % size;
        double temp[2 * MAX];
        if (rank == 0)
        {
            // tạo hàng mở rộng [A | I]
            for (int j = 0; j < N; j++)
                temp[j] = A[i][j];
            for (int j = 0; j < N; j++)
                temp[N + j] = (i == j) ? 1.0 : 0.0;
            if (dest == 0)
            {
                for (int j = 0; j < 2 * N; j++)
                    local[i / size][j] = temp[j];
            }
            else
            {
                MPI_Send(temp, 2 * N, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            if (dest == rank)
            {
                MPI_Recv(local[i / size], 2 * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Mở file CSV để ghi kết quả
    FILE *f = fopen("result.csv", "a");
    if (!f)
    {
        printf("Khong mo duoc file result.csv!\n");
        return 1;
    }

    double t1 = MPI_Wtime();
    // Bắt đầu thuật toán Gauss–Jordan song song
    for (int k = 0; k < N; k++)
    {
        // 1. Tìm pivot toàn cục
        double local_max = 0.0;
        int local_row = -1;
        for (int i = 0; i < local_rows; i++)
        {
            int g = row_index[i];
            if (g >= k)
            {
                double val = fabs(local[i][k]);
                if (val > local_max)
                {
                    local_max = val;
                    local_row = g;
                }
            }
        }

        struct
        {
            double val;
            int idx;
        } in = {local_max, local_row}, out;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        int pivot = out.idx;
        if (pivot < 0 || fabs(out.val) < EPS)
        {
            if (rank == 0)
                printf("Ma tran suy bien tai cot %d!\n", k);
            MPI_Finalize();
            return 0;
        }

        int pivot_owner = pivot % size;
        int pivot_local = pivot / size;

        int target_owner = k % size;
        int target_local = k / size;

        // 2. Hoán đổi hàng pivot <-> hàng k
        if (pivot != k)
        {
            if (rank == pivot_owner && rank == target_owner)
            {
                for (int j = 0; j < 2 * N; j++)
                {
                    double tmp = local[pivot_local][j];
                    local[pivot_local][j] = local[target_local][j];
                    local[target_local][j] = tmp;
                }
            }
            else if (rank == pivot_owner)
            {
                MPI_Send(local[pivot_local], 2 * N, MPI_DOUBLE, target_owner, 1, MPI_COMM_WORLD);
                MPI_Recv(local[pivot_local], 2 * N, MPI_DOUBLE, target_owner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else if (rank == target_owner)
            {
                double temp[2 * MAX];
                for (int j = 0; j < 2 * N; j++)
                    temp[j] = local[target_local][j];
                MPI_Recv(local[target_local], 2 * N, MPI_DOUBLE, pivot_owner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(temp, 2 * N, MPI_DOUBLE, pivot_owner, 2, MPI_COMM_WORLD);
            }
        }

        // 3. Chuẩn hóa hàng pivot và broadcast
        if (rank == target_owner)
        {
            double div = local[target_local][k];
            for (int j = 0; j < 2 * N; j++)
                local[target_local][j] /= div;
            for (int j = 0; j < 2 * N; j++)
                pivotrow[j] = local[target_local][j];
        }
        MPI_Bcast(pivotrow, 2 * N, MPI_DOUBLE, target_owner, MPI_COMM_WORLD);

        // 4. Loại bỏ phần tử cùng cột ở các hàng khác
        for (int i = 0; i < local_rows; i++)
        {
            int g = row_index[i];
            if (g == k)
                continue;
            double factor = local[i][k];
            for (int j = 0; j < 2 * N; j++)
            {
                local[i][j] -= factor * pivotrow[j];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Tiến trình 0 thu kết quả nghịch đảo
    if (rank == 0)
    {
        double inv[MAX][MAX];
        for (int r = 0; r < size; r++)
        {
            for (int i = 0; i < N; i++)
            {
                if (i % size == r)
                {
                    if (r == 0)
                    {
                        for (int j = 0; j < N; j++)
                            inv[i][j] = local[i / size][N + j];
                    }
                    else
                    {
                        double temp[MAX];
                        MPI_Recv(temp, N, MPI_DOUBLE, r, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int j = 0; j < N; j++)
                            inv[i][j] = temp[j];
                    }
                }
            }
        }
        // Kết thúc tính thời gian
        double t2 = MPI_Wtime();
        double elapsed = (double)(t2 - t1);

        // Lưu thời gian vào file csv
        fprintf(f, "MPI,%d,%.6f\n", size, elapsed);
        fclose(f);

        printf("\nMa tran nghich dao:\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
                printf("%10.6f ", inv[i][j]);
            printf("\n");
        }

        printf("\nThoi gian MPI (n=%d, p=%d): %f giay\n", N, size, elapsed);
    }
    else
    {
        for (int i = 0; i < local_rows; i++)
        {
            double temp[MAX];
            for (int j = 0; j < N; j++)
                temp[j] = local[i][N + j];
            MPI_Send(temp, N, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
