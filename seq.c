#include <stdio.h>
#include <stdlib.h>
#include <windows.h>        //Thư viện phục vụ việc tính thời gian chạy
#include <math.h>

#define MAX 300             // Dùng mảng tĩnh với cấp ma trận <=300
#define EPS 1e-12           // Dùng để so sánh với giá trị pivot

double A[MAX][MAX];         // Ma trận ban đầu
double aug[MAX][2 * MAX];   // Ma trận A|I

int main()
{
    int n; // Cấp ma trận
    printf("Nhap cap ma tran N (<= %d): ", MAX);
    scanf("%d", &n);

    int choice; // Biến lựa chọn ở switch
    printf("Nhap hay sinh ngay nhien ma tran?\n");
    printf("1.Nhap ma tran\n2.Sinh ngau nhien\n");
    printf("Chon 1 hoac 2: ");
    scanf("%d", &choice);

    switch (choice)
    {
    case 1: // Case tự nhập ma trận

        printf("Nhap ma tran A (%dx%d):\n", n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                scanf("%lf", &A[i][j]);

        // Tạo ma trận mở rộng [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                // Lấy dữ liệu từ ma trận A
                aug[i][j] = A[i][j];
            for (int j = n; j < 2 * n; j++)
                // Sinh ma trận đơn vị I
                aug[i][j] = (i == (j - n)) ? 1.0 : 0.0;
        }
        break;
    case 2: // Case sinh ma trận ngẫu nhiên

        srand(0); // Để cố định kết quả ngẫu nhiên

        // Sinh ma trận A ngẫu nhiên có đường chéo lớn (đảm bảo khả nghịch)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                // Đường chéo lớn gấp 30-100 lần phần còn lại
                aug[i][j] = (i == j) ? 100.0 : (rand() % 3 + 1);
            for (int j = 0; j < n; j++)
                // Sinh ma trận đơn vị I
                aug[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
        break;
    }
    // Mở file CSV để ghi kết quả
    FILE *f = fopen("result.csv", "w");
    if (!f)
    {
        printf("Khong mo duoc file result.csv!\n");
        return 1;
    }

    fprintf(f, "Method,Threads,Time\n");

    // Bắt đầu đo thời gian
    LARGE_INTEGER freq, t1, t2;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

    // Giải thuật Gauss - Jordan
    for (int i = 0; i < n; i++)
    {
        // 1. Tìm hàng pivot có trị tuyệt đối lớn nhất ở cột i
        int pivot_row = i;
        double max_val = fabs(aug[i][i]);

        for (int r = i + 1; r < n; r++)
        {
            double val = fabs(aug[r][i]);
            if (val > max_val)
            {
                max_val = val;
                pivot_row = r;
            }
        }

        // 2. Nếu cần, hoán đổi hàng i <-> pivot_row
        if (pivot_row != i)
        {
            for (int j = 0; j < 2 * n; j++)
            {
                double tmp = aug[i][j];
                aug[i][j] = aug[pivot_row][j];
                aug[pivot_row][j] = tmp;
            }
        }

        // 3. Kiểm tra khả nghịch
        double pivot = aug[i][i];
        if (fabs(pivot) < EPS)
        {
            printf("Ma tran khong kha nghich!\n");
            exit(0);
        }

        // 4. Chuẩn hóa hàng i
        for (int j = 0; j < 2 * n; j++)
            aug[i][j] /= pivot;

        // 5. Triệt tiêu các hàng khác
        for (int k = 0; k < n; k++)
        {
            if (k != i)
            {
                double factor = aug[k][i];
                for (int j = 0; j < 2 * n; j++)
                    aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Kết thúc đo thời gian
    QueryPerformanceCounter(&t2);
    double elapsed = (double)(t2.QuadPart - t1.QuadPart) / freq.QuadPart;

    //Lưu thời gian vào file csv
    fprintf(f, "Sequential,%d,%.6f\n", 1, elapsed);
    fclose(f);

    //Hiển thị kết quả
    int choice2;
    printf("Co in ma tran nghich dao khong?(1-Co hoac 0-Khong): ");
    scanf("%d", &choice2);

    if (choice2 == 1)
    {
        printf("\nMa tran nghich dao:\n");
        for (int i = 0; i < n; i++)
        {
            for (int j = n; j < 2 * n; j++)
                printf("%10.4f ", aug[i][j]);
            printf("\n");
        }
    }

    printf("\nThoi gian thuc thi tuan tu: %.8f giay\n", elapsed);
    return 0;
}
