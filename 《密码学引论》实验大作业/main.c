#include <gtk/gtk.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/rand.h>
#include <openssl/bn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAGIC "SM4ENC02"
#define MAGIC_LEN 8
#define IV_LEN 16
#define SM4_KEY_LEN 16
#define BUF_SIZE 4096
#define CTS_FLAG 0x01   /* bit 0 = 1 means CTS mode (plaintext >= 16 bytes) */

static GtkWidget *window;
static GtkWidget *file_entry;
static GtkWidget *log_view;

static void append_log(const char *msg) {
    GtkTextBuffer *buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(log_view));
    GtkTextIter end;
    gtk_text_buffer_get_end_iter(buffer, &end);
    gtk_text_buffer_insert(buffer, &end, msg, -1);
    gtk_text_buffer_insert(buffer, &end, "\n", -1);
}

static void write_u32(FILE *f, uint32_t v) {
    unsigned char b[4];
    b[0]=(v>>24)&0xff; b[1]=(v>>16)&0xff; b[2]=(v>>8)&0xff; b[3]=v&0xff;
    fwrite(b,1,4,f);
}

static int read_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b,1,4,f)!=4) return 0;
    *v=((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    return 1;
}

static long file_size(const char *path) {
    FILE *f=fopen(path,"rb");
    if(!f) return -1;
    fseek(f,0,SEEK_END);
    long s=ftell(f);
    fclose(f);
    return s;
}

/* ── Raw SM4 block operations (ECB mode, no chaining) ──────────────── */

static void xor_block(unsigned char *dst, const unsigned char *a, const unsigned char *b) {
    for (int i = 0; i < 16; i++)
        dst[i] = a[i] ^ b[i];
}

static int sm4_raw_encrypt_block(const unsigned char key[16],
                                 const unsigned char in[16],
                                 unsigned char out[16]) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return 0;
    int outlen = 0, ret = 1;
    unsigned char zero_iv[16] = {0};
    /* Use CBC with zero IV instead of ECB — some OpenSSL builds have buggy ECB */
    if (EVP_EncryptInit_ex(ctx, EVP_sm4_cbc(), NULL, key, zero_iv) != 1) ret = 0;
    if (ret) EVP_CIPHER_CTX_set_padding(ctx, 0);
    if (ret && EVP_EncryptUpdate(ctx, out, &outlen, in, 16) != 1) ret = 0;
    if (ret && outlen != 16) ret = 0;
    EVP_CIPHER_CTX_free(ctx);
    return ret;
}

static int sm4_raw_decrypt_block(const unsigned char key[16],
                                 const unsigned char in[16],
                                 unsigned char out[16]) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return 0;
    int outlen = 0, ret = 1;
    unsigned char zero_iv[16] = {0};
    /* Use CBC with zero IV instead of ECB — some OpenSSL builds have buggy ECB */
    if (EVP_DecryptInit_ex(ctx, EVP_sm4_cbc(), NULL, key, zero_iv) != 1) ret = 0;
    if (ret) EVP_CIPHER_CTX_set_padding(ctx, 0);
    if (ret && EVP_DecryptUpdate(ctx, out, &outlen, in, 16) != 1) ret = 0;
    if (ret && outlen != 16) ret = 0;
    EVP_CIPHER_CTX_free(ctx);
    return ret;
}

/* ── RSA key generation & protection ──────────────────────────────── */

static int generate_rsa_keys(const char *pub_path, const char *pri_path, char *err, size_t errlen) {
    int ret = 0;
    RSA *rsa = NULL;
    BIGNUM *bn = NULL;
    FILE *fp = NULL;

    bn = BN_new();
    rsa = RSA_new();
    if (!bn || !rsa) { snprintf(err, errlen, "创建RSA对象失败"); goto end; }
    if (!BN_set_word(bn, RSA_F4)) { snprintf(err, errlen, "设置指数失败"); goto end; }
    if (!RSA_generate_key_ex(rsa, 2048, bn, NULL)) { snprintf(err, errlen, "生成RSA密钥失败"); goto end; }

    fp = fopen(pri_path, "wb");
    if (!fp) { snprintf(err, errlen, "无法写入私钥文件"); goto end; }
    if (!PEM_write_RSAPrivateKey(fp, rsa, NULL, NULL, 0, NULL, NULL)) { snprintf(err, errlen, "写入私钥失败"); goto end; }
    fclose(fp); fp = NULL;

    fp = fopen(pub_path, "wb");
    if (!fp) { snprintf(err, errlen, "无法写入公钥文件"); goto end; }
    if (!PEM_write_RSAPublicKey(fp, rsa)) { snprintf(err, errlen, "写入公钥失败"); goto end; }
    fclose(fp); fp = NULL;

    ret = 1;
end:
    if (fp) fclose(fp);
    if (rsa) RSA_free(rsa);
    if (bn) BN_free(bn);
    return ret;
}

static int rsa_encrypt_key(const char *pub_path, const unsigned char *key, int key_len,
                           unsigned char **out, int *out_len, char *err, size_t errlen) {
    FILE *fp = fopen(pub_path, "rb");
    if (!fp) { snprintf(err, errlen, "找不到 public.pem，请先生成RSA密钥对"); return 0; }
    RSA *rsa = PEM_read_RSAPublicKey(fp, NULL, NULL, NULL);
    fclose(fp);
    if (!rsa) { snprintf(err, errlen, "读取公钥失败"); return 0; }
    int size = RSA_size(rsa);
    *out = (unsigned char*)malloc(size);
    if (!*out) { RSA_free(rsa); snprintf(err, errlen, "内存不足"); return 0; }
    int n = RSA_public_encrypt(key_len, key, *out, rsa, RSA_PKCS1_OAEP_PADDING);
    RSA_free(rsa);
    if (n <= 0) { free(*out); *out=NULL; snprintf(err, errlen, "RSA加密SM4密钥失败"); return 0; }
    *out_len = n;
    return 1;
}

static int rsa_decrypt_key(const char *pri_path, const unsigned char *enc_key, int enc_key_len,
                           unsigned char *key, int *key_len, char *err, size_t errlen) {
    FILE *fp = fopen(pri_path, "rb");
    if (!fp) { snprintf(err, errlen, "找不到 private.pem，无法解密SM4密钥"); return 0; }
    RSA *rsa = PEM_read_RSAPrivateKey(fp, NULL, NULL, NULL);
    fclose(fp);
    if (!rsa) { snprintf(err, errlen, "读取私钥失败"); return 0; }
    int n = RSA_private_decrypt(enc_key_len, enc_key, key, rsa, RSA_PKCS1_OAEP_PADDING);
    RSA_free(rsa);
    if (n != SM4_KEY_LEN) { snprintf(err, errlen, "RSA解密SM4密钥失败"); return 0; }
    *key_len = n;
    return 1;
}

/* ── File encryption with CBC-CTS (Ciphertext Stealing) ───────────── */
/*
 * CTS variant: CBC-CS1 (NIST SP 800-38A Addendum).
 * For plaintext >= 16 bytes, ciphertext length == plaintext length (zero expansion).
 * For plaintext <  16 bytes, fall back to PKCS#7 padding (16-byte ciphertext).
 *
 * File format:
 *   MAGIC(8) | enc_key_len(4) | iv_len(4) | enc_key(var) | IV(16) | flags(1) | ciphertext
 *
 * flags & 0x01 = CTS mode (plaintext >= 16)
 * flags      0 = PKCS#7 mode (plaintext < 16, or backward-compatible)
 */

static int encrypt_file(const char *in_path, const char *out_path, const char *pub_path,
                        double *seconds, double *mbps, char *err, size_t errlen) {
    FILE *fin = NULL, *fout = NULL;
    EVP_CIPHER_CTX *ctx = NULL;
    unsigned char sm4_key[SM4_KEY_LEN], iv[IV_LEN];
    unsigned char *enc_key = NULL;
    int enc_key_len = 0;
    unsigned char inbuf[BUF_SIZE], outbuf[BUF_SIZE + EVP_MAX_BLOCK_LENGTH];
    int outlen = 0, tmplen = 0;
    size_t nread;
    long input_size = file_size(in_path);
    clock_t start, end;

    if (input_size < 0) { snprintf(err, errlen, "无法读取输入文件"); return 0; }
    if (RAND_bytes(sm4_key, SM4_KEY_LEN) != 1 || RAND_bytes(iv, IV_LEN) != 1) {
        snprintf(err, errlen, "随机生成SM4密钥/IV失败"); return 0;
    }
    if (!rsa_encrypt_key(pub_path, sm4_key, SM4_KEY_LEN, &enc_key, &enc_key_len, err, errlen))
        return 0;

    fin = fopen(in_path, "rb");
    fout = fopen(out_path, "wb");
    if (!fin || !fout) { snprintf(err, errlen, "打开输入/输出文件失败"); goto fail; }

    /* ── Write header ── */
    fwrite(MAGIC, 1, MAGIC_LEN, fout);
    write_u32(fout, (uint32_t)enc_key_len);
    write_u32(fout, (uint32_t)IV_LEN);
    fwrite(enc_key, 1, enc_key_len, fout);
    fwrite(iv, 1, IV_LEN, fout);

    start = clock();

    /* ── Empty file ── */
    if (input_size == 0) {
        unsigned char flags = 0;
        fwrite(&flags, 1, 1, fout);
        goto done_enc;
    }

    /* ── Plaintext < 16 bytes: PKCS#7 fallback ── */
    if (input_size < 16) {
        unsigned char flags = 0;
        fwrite(&flags, 1, 1, fout);

        ctx = EVP_CIPHER_CTX_new();
        if (!ctx) { snprintf(err, errlen, "创建加密上下文失败"); goto fail; }
        if (EVP_EncryptInit_ex(ctx, EVP_sm4_cbc(), NULL, sm4_key, iv) != 1) {
            snprintf(err, errlen, "初始化SM4-CBC加密失败"); goto fail;
        }
        while ((nread = fread(inbuf, 1, BUF_SIZE, fin)) > 0) {
            if (EVP_EncryptUpdate(ctx, outbuf, &outlen, inbuf, (int)nread) != 1) {
                snprintf(err, errlen, "加密过程失败"); goto fail;
            }
            fwrite(outbuf, 1, outlen, fout);
        }
        if (EVP_EncryptFinal_ex(ctx, outbuf, &tmplen) != 1) {
            snprintf(err, errlen, "加密收尾失败"); goto fail;
        }
        fwrite(outbuf, 1, tmplen, fout);
        EVP_CIPHER_CTX_free(ctx);
        ctx = NULL;
        goto done_enc;
    }

    /* ── Plaintext >= 16 bytes: CBC-CTS mode ── */
    {
        unsigned char flags = CTS_FLAG;
        fwrite(&flags, 1, 1, fout);

        int q = (int)(input_size / 16);   /* number of full 16-byte blocks */
        int r = (int)(input_size % 16);   /* remaining bytes in last partial block */

        /* Create EVP context with padding DISABLED for standard CBC streaming */
        ctx = EVP_CIPHER_CTX_new();
        if (!ctx) { snprintf(err, errlen, "创建加密上下文失败"); goto fail; }
        if (EVP_EncryptInit_ex(ctx, EVP_sm4_cbc(), NULL, sm4_key, iv) != 1) {
            snprintf(err, errlen, "初始化SM4-CBC加密失败"); goto fail;
        }
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        /* Track last ciphertext block for CTS (initialized to IV) */
        unsigned char prev_ct[16];
        memcpy(prev_ct, iv, 16);

        /*
         * Phase 1: Stream standard CBC for all blocks except the last 1-2.
         *   r == 0  →  process all q blocks (standard CBC, no CTS needed)
         *   q == 1  →  process 0 blocks (only one full block, handle in CTS Phase 2)
         *   q >= 2, r > 0 → process q-1 blocks (leave last full + partial for CTS)
         */
        long cbc_bytes;
        if (r == 0) {
            cbc_bytes = input_size;            /* all blocks, standard CBC */
        } else if (q == 1) {
            cbc_bytes = 0;                     /* nothing to stream */
        } else {
            cbc_bytes = (long)(q - 1) * 16;    /* q-1 full blocks */
        }

        long remaining = cbc_bytes;
        while (remaining > 0) {
            int chunk = (remaining > BUF_SIZE) ? BUF_SIZE : (int)remaining;
            nread = fread(inbuf, 1, (size_t)chunk, fin);
            if (nread <= 0) break;
            if (EVP_EncryptUpdate(ctx, outbuf, &outlen, inbuf, (int)nread) != 1) {
                snprintf(err, errlen, "加密过程失败"); goto fail;
            }
            fwrite(outbuf, 1, outlen, fout);
            if (outlen >= 16)
                memcpy(prev_ct, outbuf + outlen - 16, 16);
            remaining -= (long)nread;
        }

        if (r == 0) {
            /* Standard CBC: finalize */
            if (EVP_EncryptFinal_ex(ctx, outbuf, &tmplen) != 1) {
                snprintf(err, errlen, "加密收尾失败"); goto fail;
            }
            fwrite(outbuf, 1, tmplen, fout);
        } else {
            /*
             * Phase 2: CTS tail (CBC-CS1).
             *
             * Pq  = last full plaintext block (16 bytes)
             * P*  = partial plaintext block (r bytes, 1 <= r <= 15)
             *
             * Step 1: C_temp = E_k(Pq XOR prev_ct)          [encrypt penultimate block]
             * Step 2: P_padded = P* || C_temp[0..15-r]     [steal first 16-r bytes of C_temp]
             * Step 3: C_last  = E_k(P_padded XOR C_temp)
             * Step 4: Output C_last (16 bytes) || C_temp[0..r-1] (r bytes)
             */
            unsigned char Pq[16], Pr[16] = {0};
            if (fread(Pq, 1, 16, fin) != 16) {
                snprintf(err, errlen, "读取倒数第二个明文块失败"); goto fail;
            }
            if (fread(Pr, 1, (size_t)r, fin) != (size_t)r) {
                snprintf(err, errlen, "读取末尾部分明文块失败"); goto fail;
            }

            unsigned char xor_buf[16], C_temp[16], C_last[16];

            /* C_temp = E_k(Pq XOR prev_ct) */
            xor_block(xor_buf, Pq, prev_ct);
            if (!sm4_raw_encrypt_block(sm4_key, xor_buf, C_temp)) {
                snprintf(err, errlen, "SM4块加密(C_temp)失败"); goto fail;
            }

            /* P_padded = P* || C_temp[0..15-r] */
            unsigned char P_padded[16];
            memcpy(P_padded, Pr, (size_t)r);
            memcpy(P_padded + r, C_temp, (size_t)(16 - r));

            /* C_last = E_k(P_padded XOR C_temp) */
            xor_block(xor_buf, P_padded, C_temp);
            if (!sm4_raw_encrypt_block(sm4_key, xor_buf, C_last)) {
                snprintf(err, errlen, "SM4块加密(C_last)失败"); goto fail;
            }

            /* Output: C_last (full 16 bytes) || C_temp[0..r-1] (r bytes) */
            fwrite(C_last, 1, 16, fout);
            fwrite(C_temp, 1, (size_t)r, fout);
        }
    }

done_enc:
    end = clock();
    *seconds = (double)(end - start) / CLOCKS_PER_SEC;
    if (*seconds <= 0) *seconds = 0.000001;
    *mbps = ((double)input_size / 1024.0 / 1024.0) / (*seconds);

    fclose(fin);
    fclose(fout);
    if (ctx) EVP_CIPHER_CTX_free(ctx);
    free(enc_key);
    return 1;

fail:
    if (fin) fclose(fin);
    if (fout) fclose(fout);
    if (ctx) EVP_CIPHER_CTX_free(ctx);
    free(enc_key);
    return 0;
}

/* ── File decryption with CBC-CTS ─────────────────────────────────── */

static int decrypt_file(const char *in_path, const char *out_path, const char *pri_path,
                        double *seconds, double *mbps, char *err, size_t errlen) {
    FILE *fin = NULL, *fout = NULL;
    EVP_CIPHER_CTX *ctx = NULL;
    char magic[MAGIC_LEN];
    uint32_t enc_key_len = 0, iv_len = 0;
    unsigned char *enc_key = NULL;
    unsigned char sm4_key[SM4_KEY_LEN], iv[IV_LEN];
    int sm4_key_len = 0;
    unsigned char inbuf[BUF_SIZE], outbuf[BUF_SIZE + EVP_MAX_BLOCK_LENGTH];
    int outlen = 0, tmplen = 0;
    size_t nread;
    long file_size_val = file_size(in_path);
    unsigned char flags = 0;
    clock_t start, end;

    fin = fopen(in_path, "rb");
    fout = fopen(out_path, "wb");
    if (!fin || !fout) { snprintf(err, errlen, "打开输入/输出文件失败"); goto fail; }

    /* ── Read & validate header ── */
    if (fread(magic, 1, MAGIC_LEN, fin) != MAGIC_LEN ||
        memcmp(magic, MAGIC, MAGIC_LEN) != 0) {
        snprintf(err, errlen, "文件格式不正确，不是本程序生成的.enc文件"); goto fail;
    }
    if (!read_u32(fin, &enc_key_len) || !read_u32(fin, &iv_len) ||
        iv_len != IV_LEN || enc_key_len == 0 || enc_key_len > 4096) {
        snprintf(err, errlen, "文件头损坏"); goto fail;
    }
    enc_key = (unsigned char*)malloc(enc_key_len);
    if (!enc_key) { snprintf(err, errlen, "内存不足"); goto fail; }
    if (fread(enc_key, 1, enc_key_len, fin) != enc_key_len ||
        fread(iv, 1, IV_LEN, fin) != IV_LEN) {
        snprintf(err, errlen, "读取文件头失败"); goto fail;
    }
    if (!rsa_decrypt_key(pri_path, enc_key, enc_key_len, sm4_key, &sm4_key_len, err, errlen))
        goto fail;

    /* ── Read flags byte ── */
    if (fread(&flags, 1, 1, fin) != 1) {
        /* Backward compatibility: old format without flags byte, assume PKCS#7 */
        snprintf(err, errlen, "文件格式不兼容（缺少模式标识），请使用新版程序加密");
        goto fail;
    }

    /* Compute ciphertext size */
    long header_size = MAGIC_LEN + 4 + 4 + (long)enc_key_len + IV_LEN + 1; /* +1 for flags */
    long ct_size = file_size_val - header_size;
    if (ct_size < 0) { snprintf(err, errlen, "密文数据不足"); goto fail; }

    start = clock();

    /* ── Empty plaintext ── */
    if (ct_size == 0) {
        goto done_dec;
    }

    /* ── PKCS#7 mode (plaintext was < 16 bytes) ── */
    if (!(flags & CTS_FLAG)) {
        ctx = EVP_CIPHER_CTX_new();
        if (!ctx) { snprintf(err, errlen, "创建解密上下文失败"); goto fail; }
        if (EVP_DecryptInit_ex(ctx, EVP_sm4_cbc(), NULL, sm4_key, iv) != 1) {
            snprintf(err, errlen, "初始化SM4-CBC解密失败"); goto fail;
        }
        while ((nread = fread(inbuf, 1, BUF_SIZE, fin)) > 0) {
            if (EVP_DecryptUpdate(ctx, outbuf, &outlen, inbuf, (int)nread) != 1) {
                snprintf(err, errlen, "解密过程失败"); goto fail;
            }
            fwrite(outbuf, 1, outlen, fout);
        }
        if (EVP_DecryptFinal_ex(ctx, outbuf, &tmplen) != 1) {
            snprintf(err, errlen, "解密失败：文件损坏或私钥不匹配"); goto fail;
        }
        fwrite(outbuf, 1, tmplen, fout);
        EVP_CIPHER_CTX_free(ctx);
        ctx = NULL;
        goto done_dec;
    }

    /* ── CTS mode (flags & CTS_FLAG) ── */
    {
        int r = (int)(ct_size % 16);

        ctx = EVP_CIPHER_CTX_new();
        if (!ctx) { snprintf(err, errlen, "创建解密上下文失败"); goto fail; }
        if (EVP_DecryptInit_ex(ctx, EVP_sm4_cbc(), NULL, sm4_key, iv) != 1) {
            snprintf(err, errlen, "初始化SM4-CBC解密失败"); goto fail;
        }
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        /* Track the last READ ciphertext block (used as "previous CT" for CBC chain recovery) */
        unsigned char prev_ct[16];
        memcpy(prev_ct, iv, 16);

        if (r == 0) {
            /* Standard CBC (no CTS): all blocks are multiples of 16 */
            long remaining = ct_size;
            while (remaining > 0) {
                int chunk = (remaining > BUF_SIZE) ? BUF_SIZE : (int)remaining;
                nread = fread(inbuf, 1, (size_t)chunk, fin);
                if (nread <= 0) break;
                if (EVP_DecryptUpdate(ctx, outbuf, &outlen, inbuf, (int)nread) != 1) {
                    snprintf(err, errlen, "解密过程失败"); goto fail;
                }
                fwrite(outbuf, 1, outlen, fout);
                remaining -= (long)nread;
            }
            if (EVP_DecryptFinal_ex(ctx, outbuf, &tmplen) != 1) {
                snprintf(err, errlen, "解密失败：文件损坏或私钥不匹配"); goto fail;
            }
            fwrite(outbuf, 1, tmplen, fout);
        } else {
            /*
             * CTS decryption (CBC-CS1).
             *
             * Ciphertext layout:
             *   C1 || C2 || ... || C_{q-2} || C_last(16) || C_temp_star(r)
             *
             * Phase 1: Standard CBC decrypt for first ct_size - 16 - r bytes.
             * Phase 2: CTS tail recovery.
             */
            long cbc_bytes = ct_size - 16 - (long)r;

            /* Phase 1: Standard CBC decryption */
            long remaining = cbc_bytes;
            while (remaining > 0) {
                int chunk = (remaining > BUF_SIZE) ? BUF_SIZE : (int)remaining;
                nread = fread(inbuf, 1, (size_t)chunk, fin);
                if (nread <= 0) break;
                if (EVP_DecryptUpdate(ctx, outbuf, &outlen, inbuf, (int)nread) != 1) {
                    snprintf(err, errlen, "解密过程失败"); goto fail;
                }
                fwrite(outbuf, 1, outlen, fout);
                /* Track last ciphertext block read */
                if (nread >= 16)
                    memcpy(prev_ct, inbuf + nread - 16, 16);
                remaining -= (long)nread;
            }

            /* Phase 2: CTS tail decryption (CBC-CS1) */
            unsigned char C_last[16], C_temp_star[16] = {0};

            if (fread(C_last, 1, 16, fin) != 16) {
                snprintf(err, errlen, "读取密文CTS交换块(C_last)失败"); goto fail;
            }
            if (fread(C_temp_star, 1, (size_t)r, fin) != (size_t)r) {
                snprintf(err, errlen, "读取密文CTS尾部(C_temp*)失败"); goto fail;
            }

            /*
             * Step 1: D_last = D_k(C_last)
             *         D_last[i] = P_padded[i] XOR C_temp[i]  for i=0..15
             *   where P_padded = P* || C_temp[0..15-r]
             *   so    D_last[0..r-1] = P*[i] XOR C_temp[i]
             *   and   D_last[i] = C_temp[i-r] XOR C_temp[i]  for i=r..15
             */
            unsigned char D_last[16];
            if (!sm4_raw_decrypt_block(sm4_key, C_last, D_last)) {
                snprintf(err, errlen, "SM4块解密(C_last)失败"); goto fail;
            }

            /*
             * Step 2: Recover partial plaintext P*
             *   P*[i] = D_last[i] XOR C_temp_star[i]   (i = 0..r-1)
             */
            unsigned char P_star[16] = {0};
            for (int i = 0; i < r; i++) {
                P_star[i] = D_last[i] ^ C_temp_star[i];
            }

            /*
             * Step 3: Reconstruct full C_temp[0..15]
             *   C_temp[0..r-1] = C_temp_star (from ciphertext)
             *   C_temp[i] = D_last[i] XOR C_temp[i-r]  for i=r..15
             * (This is solvable sequentially because C_temp[i-r] is always
             *  either known from C_temp_star or previously recovered.)
             */
            unsigned char C_temp_full[16];
            memcpy(C_temp_full, C_temp_star, (size_t)r);
            for (int i = r; i < 16; i++) {
                C_temp_full[i] = D_last[i] ^ C_temp_full[i - r];
            }

            /*
             * Step 4: Decrypt penultimate block
             *   Pq = D_k(C_temp_full) XOR prev_ct
             */
            unsigned char D_temp[16];
            if (!sm4_raw_decrypt_block(sm4_key, C_temp_full, D_temp)) {
                snprintf(err, errlen, "SM4块解密(C_temp)失败"); goto fail;
            }

            unsigned char Pq[16];
            xor_block(Pq, D_temp, prev_ct);

            /* Output: Pq (full 16 bytes) || P_star (r bytes) */
            fwrite(Pq, 1, 16, fout);
            fwrite(P_star, 1, (size_t)r, fout);
        }
    }

done_dec:
    end = clock();
    *seconds = (double)(end - start) / CLOCKS_PER_SEC;
    if (*seconds <= 0) *seconds = 0.000001;
    *mbps = ((double)ct_size / 1024.0 / 1024.0) / (*seconds);

    fclose(fin);
    fclose(fout);
    if (ctx) EVP_CIPHER_CTX_free(ctx);
    free(enc_key);
    return 1;

fail:
    if (fin) fclose(fin);
    if (fout) fclose(fout);
    if (ctx) EVP_CIPHER_CTX_free(ctx);
    free(enc_key);
    return 0;
}

/* ── GUI helpers ──────────────────────────────────────────────────── */

static char *make_out_path(const char *input, const char *suffix) {
    size_t len = strlen(input) + strlen(suffix) + 1;
    char *out = (char*)malloc(len);
    snprintf(out, len, "%s%s", input, suffix);
    return out;
}

static void on_choose_file(GtkWidget *widget, gpointer data) {
    (void)widget; (void)data;
    GtkWidget *dialog = gtk_file_chooser_dialog_new("选择文件", GTK_WINDOW(window),
        GTK_FILE_CHOOSER_ACTION_OPEN,
        "取消", GTK_RESPONSE_CANCEL,
        "选择", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
        char *filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        gtk_entry_set_text(GTK_ENTRY(file_entry), filename);
        g_free(filename);
    }
    gtk_widget_destroy(dialog);
}

static void on_genkey(GtkWidget *widget, gpointer data) {
    (void)widget; (void)data;
    char err[256] = {0};
    if (generate_rsa_keys("public.pem", "private.pem", err, sizeof(err))) {
        append_log("RSA密钥对生成成功：public.pem / private.pem");
    } else {
        append_log(err);
    }
}

static void on_encrypt(GtkWidget *widget, gpointer data) {
    (void)widget; (void)data;
    const char *input = gtk_entry_get_text(GTK_ENTRY(file_entry));
    if (!input || strlen(input) == 0) { append_log("请先选择要加密的文件"); return; }
    char *output = make_out_path(input, ".enc");
    char err[256] = {0}, msg[512];
    double sec = 0, speed = 0;
    if (encrypt_file(input, output, "public.pem", &sec, &speed, err, sizeof(err))) {
        long isz = file_size(input);
        long osz = file_size(output);
        snprintf(msg, sizeof(msg),
            "加密成功\n"
            "输出文件：%s\n"
            "明文大小：%ld 字节  密文大小：%ld 字节\n"
            "耗时：%.6f 秒\n"
            "速度：%.3f MB/s",
            output, isz, osz, sec, speed);
        append_log(msg);
    } else {
        append_log(err);
    }
    free(output);
}

static void on_decrypt(GtkWidget *widget, gpointer data) {
    (void)widget; (void)data;
    const char *input = gtk_entry_get_text(GTK_ENTRY(file_entry));
    if (!input || strlen(input) == 0) { append_log("请先选择要解密的.enc文件"); return; }
    char *output = make_out_path(input, ".dec");
    char err[256] = {0}, msg[512];
    double sec = 0, speed = 0;
    if (decrypt_file(input, output, "private.pem", &sec, &speed, err, sizeof(err))) {
        long isz = file_size(input);
        long osz = file_size(output);
        snprintf(msg, sizeof(msg),
            "解密成功\n"
            "输出文件：%s\n"
            "密文大小：%ld 字节  明文大小：%ld 字节\n"
            "耗时：%.6f 秒\n"
            "速度：%.3f MB/s",
            output, isz, osz, sec, speed);
        append_log(msg);
    } else {
        append_log(err);
    }
    free(output);
}

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);
    OpenSSL_add_all_algorithms();

    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "SM4文件加密系统 - CBC-CTS模式");
    gtk_window_set_default_size(GTK_WINDOW(window), 720, 520);
    gtk_container_set_border_width(GTK_CONTAINER(window), 15);
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    GtkWidget *title = gtk_label_new("SM4 文件加密/解密系统 (CBC-CTS)");
    gtk_box_pack_start(GTK_BOX(vbox), title, FALSE, FALSE, 5);

    GtkWidget *subtitle = gtk_label_new("短块处理：密文挪用（Ciphertext Stealing）+ 密文链接（CBC）");
    gtk_box_pack_start(GTK_BOX(vbox), subtitle, FALSE, FALSE, 0);

    GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(GTK_BOX(vbox), hbox, FALSE, FALSE, 5);
    file_entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(file_entry), "请选择文件路径");
    gtk_box_pack_start(GTK_BOX(hbox), file_entry, TRUE, TRUE, 0);
    GtkWidget *choose_btn = gtk_button_new_with_label("选择文件");
    gtk_box_pack_start(GTK_BOX(hbox), choose_btn, FALSE, FALSE, 0);
    g_signal_connect(choose_btn, "clicked", G_CALLBACK(on_choose_file), NULL);

    GtkWidget *btn_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(GTK_BOX(vbox), btn_box, FALSE, FALSE, 5);
    GtkWidget *gen_btn = gtk_button_new_with_label("生成RSA密钥对");
    GtkWidget *enc_btn = gtk_button_new_with_label("加密文件");
    GtkWidget *dec_btn = gtk_button_new_with_label("解密文件");
    gtk_box_pack_start(GTK_BOX(btn_box), gen_btn, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(btn_box), enc_btn, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(btn_box), dec_btn, TRUE, TRUE, 0);
    g_signal_connect(gen_btn, "clicked", G_CALLBACK(on_genkey), NULL);
    g_signal_connect(enc_btn, "clicked", G_CALLBACK(on_encrypt), NULL);
    g_signal_connect(dec_btn, "clicked", G_CALLBACK(on_decrypt), NULL);

    GtkWidget *scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_box_pack_start(GTK_BOX(vbox), scroll, TRUE, TRUE, 5);
    log_view = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(log_view), FALSE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(log_view), GTK_WRAP_WORD_CHAR);
    gtk_container_add(GTK_CONTAINER(scroll), log_view);

    append_log("使用步骤：1. 生成RSA密钥对  2. 选择文件  3. 加密或解密");
    append_log("加密算法：SM4-CBC-CTS（密文挪用短块处理）");
    append_log("密钥保护：RSA-2048-OAEP");
    append_log("短块处理：明文≥16字节时采用CBC-CTS密文挪用，密文长度=明文长度\n"
               "          明文<16字节时采用PKCS#7填充（仅此情况有1~15字节膨胀）\n");

    gtk_widget_show_all(window);
    gtk_main();
    return 0;
}
