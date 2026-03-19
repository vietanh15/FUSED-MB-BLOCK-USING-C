#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#define OUTPUT_F 64
#define INPUT_C 128

int main() {
    FILE* f = fopen("weights_pwc.txt", "w");
    if (!f) {
        printf("Loi: Khong the tao file weights_pwc.txt\n");
        return 1;
    }
    srand((unsigned int)time(NULL));
    for (int of = 0; of < OUTPUT_F; of++) {
        for (int ic = 0; ic < INPUT_C; ic++) {
            int8_t w = (rand() % 256) - 128; // Giá trị int8 [-128, 127]
            fprintf(f, "%d\n", w);
        }
    }
    fclose(f);
    printf("Da tao file weights_pwc.txt!\n");
    return 0;
}
