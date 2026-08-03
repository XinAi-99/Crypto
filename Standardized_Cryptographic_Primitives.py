import hmac

from Cryptodome.Random import get_random_bytes
from ecpy.curves import Curve, Point
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import math
import time
from decimal import Decimal, getcontext
from hashlib import shake_256
import secrets
import hashlib
from fuzzyextractor import FuzzyExtractor
from pypuf.simulation import ArbiterPUF
import numpy as np
import pysodium
import ascon
from Cryptodome.Cipher import AES
import random
from kyber_py.kyber import Kyber512
import struct
#########################################################################################################################################
def Generate_Nonce() -> bytes:

    bytes_len = 16  # 128 bits = 16 bytes

    nonce = secrets.token_bytes(bytes_len)

    if len(nonce) != bytes_len:
        raise RuntimeError("unexpected RNG output length / 随机输出长度异常")

    return nonce


def Xor_Data(*data: bytes) -> bytes:

    if not data:
        raise ValueError("at least one bytes input is required")

    # 计算最大长度
    max_len = max(len(d) for d in data)

    # 将所有数据左侧补 0（高位补 0），以对齐最大长度
    padded_data = [
        d.rjust(max_len, b'\x00') for d in data
    ]

    # 按字节 XOR
    result = bytearray(max_len)
    for i in range(max_len):
        byte = 0
        for d in padded_data:
            byte ^= d[i]
        result[i] = byte

    return bytes(result)

def Concat_Data(*data: bytes) -> bytes:

    if not data:
        return b''

    return b''.join(data)

def Recover_Data(
    xor_result: bytes,
    output_length: int,
    *known_data: bytes
) -> bytes:

    if not isinstance(xor_result, (bytes, bytearray)):
        raise TypeError("xor_result must be bytes-like")

    if output_length <= 0:
        raise ValueError("output_length must be a positive integer")

    # 将 xor_result 作为初始恢复值
    max_len = max(
        [len(xor_result)] + [len(d) for d in known_data]
    )

    # 左侧补 0，对齐长度（等价于原 zfill 行为）
    result = xor_result.rjust(max_len, b'\x00')

    # 依次异或已知数据
    for known in known_data:
        known_padded = known.rjust(max_len, b'\x00')
        result = bytes(a ^ b for a, b in zip(result, known_padded))

    # 高位补 0 或截断，确保输出长度正确
    if len(result) < output_length:
        result = result.rjust(output_length, b'\x00')
    elif len(result) > output_length:
        result = result[-output_length:]  # 保留低位（等价于你原来的 [-output_length:]）

    return result

def Split_Data(data: bytes, *lengths: int) -> list[bytes]:

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes-like")

    if any(length <= 0 for length in lengths):
        raise ValueError("all lengths must be positive integers")

    total_length = sum(lengths)

    if len(data) < total_length:
        raise ValueError("data length is insufficient for splitting")

    result = []
    start = 0

    for length in lengths:
        result.append(data[start:start + length])
        start += length

    return result

#########################################################################################################################################
#Hash
def Hash(data: bytes) -> bytes:

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like (bytes/bytearray/memoryview)")

    # memoryview/bytearray 统一转为 bytes，确保稳定行为
    return hashlib.sha256(bytes(data)).digest()

#########################################################################################################################################
#PUF
_PUF_CACHE = {}

def _get_puf(device_seed: int, noisiness: float) -> ArbiterPUF:
    """
    复用同一台“物理设备”实例。
    说明：pypuf 的噪声由实例内部的 RNG 控制；
    复用同一实例即可确保每次 eval 的噪声抽样不同。
    """
    key = (int(device_seed), float(noisiness))
    if key not in _PUF_CACHE:
        _PUF_CACHE[key] = ArbiterPUF(
            n=128,
            seed=device_seed,
            noisiness=noisiness
        )
    return _PUF_CACHE[key]


def Noisy_PUF(challenge: bytes, device_seed: int = 0, noisiness: float = 0.05) -> bytes:
    if not isinstance(challenge, (bytes, bytearray, memoryview)):
        raise TypeError("challenge must be bytes-like")
    challenge = bytes(challenge)
    if len(challenge) != 16:
        raise ValueError("challenge must be exactly 16 bytes (128 bits)")

    shake = shake_256(challenge)
    raw = shake.digest((128 * 128 + 7) // 8)  # 向上取整到字节
    bits = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8),
        bitorder="big"
    )[:128 * 128]

    challenges01 = bits.reshape(128, 128).astype(np.int8)

    challenges_pm = challenges01 * 2 - 1

    puf = _get_puf(device_seed, noisiness)

    responses_pm = puf.eval(challenges_pm)  # shape: (128,)

    resp_bits = ((responses_pm + 1) // 2).astype(np.uint8)  # (128,)
    response = np.packbits(resp_bits, bitorder="big").tobytes()

    if len(response) != 16:
        raise RuntimeError("unexpected PUF response length")

    return response

def Ideal_PUF(challenge: bytes, device_seed: int = 0, noisiness: float = 0) -> bytes:
    if not isinstance(challenge, (bytes, bytearray, memoryview)):
        raise TypeError("challenge must be bytes-like")
    challenge = bytes(challenge)
    if len(challenge) != 16:
        raise ValueError("challenge must be exactly 16 bytes (128 bits)")

    shake = shake_256(challenge)
    raw = shake.digest((128 * 128 + 7) // 8)
    bits = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8),
        bitorder="big"
    )[:128 * 128]

    challenges01 = bits.reshape(128, 128).astype(np.int8)

    challenges_pm = challenges01 * 2 - 1

    puf = _get_puf(device_seed, noisiness)

    responses_pm = puf.eval(challenges_pm)  # shape: (128,)

    resp_bits = ((responses_pm + 1) // 2).astype(np.uint8)  # (128,)
    response = np.packbits(resp_bits, bitorder="big").tobytes()

    # 防御性检查
    if len(response) != 16:
        raise RuntimeError("unexpected PUF response length")

    return response

#########################################################################################################################################
#Fuzzy extractor
def Gen(biometric: bytes) -> tuple[bytes, bytes]:

    if not isinstance(biometric, (bytes, bytearray, memoryview)):
        raise TypeError("biometric must be bytes-like")
    biometric = bytes(biometric)
    if len(biometric) != 16:
        raise ValueError("biometric must be exactly 16 bytes (128 bits)")


    salt = secrets.token_bytes(2)


    key = hashlib.sha256(biometric + salt).digest()  # 32B


    mask = hashlib.sha256(salt).digest()[:16]


    codeword = bytes(a ^ b for a, b in zip(biometric, mask))


    helper = salt + codeword  # 2B + 16B = 18B

    return key, helper

def Rep(noisy_biometric: bytes, helper: bytes) -> bytes:

    if not isinstance(noisy_biometric, (bytes, bytearray, memoryview)):
        raise TypeError("noisy_biometric must be bytes-like")
    noisy_biometric = bytes(noisy_biometric)
    if len(noisy_biometric) != 16:
        raise ValueError("noisy_biometric must be exactly 16 bytes (128 bits)")

    if not isinstance(helper, (bytes, bytearray, memoryview)):
        raise TypeError("helper must be bytes-like")
    helper = bytes(helper)
    if len(helper) != 18:
        raise ValueError("helper must be exactly 18 bytes (144 bits)")

    salt, codeword = helper[:2], helper[2:]  # salt(2B) || codeword(16B)


    mask = hashlib.sha256(salt).digest()[:16]


    ref_bio = bytes(a ^ b for a, b in zip(codeword, mask))

    diff = bytes(a ^ b for a, b in zip(noisy_biometric, ref_bio))
    distance = bin(int.from_bytes(diff, "big")).count("1")

    t = 8
    if distance > t:
        raise ValueError("reconstruction failed: out of tolerance")

    # 返回原始密钥 = SHA-256(ref_bio || salt)
    return hashlib.sha256(ref_bio + salt).digest()

#########################################################################################################################################
#AEAD
#AEGIS-128L
def AEGIS128_Encrypt(
    plaintext: bytes,
    associated_data: bytes,
    nonce: bytes,
    key: bytes
) -> bytes:

    # AEGIS-128L AEAD 加密
    ciphertext = pysodium.crypto_aead_aegis128l_encrypt(
        plaintext,
        associated_data,
        nonce,
        key
    )

    return ciphertext

def AEGIS128_Decrypt(
    ciphertext: bytes,
    associated_data: bytes,
    nonce: bytes,
    key: bytes
) -> bytes:

    try:
        plaintext = pysodium.crypto_aead_aegis128l_decrypt(
            ciphertext,
            associated_data,
            nonce,
            key
        )
    except ValueError as e:
        raise ValueError(
            "Decryption failed: authentication tag invalid or data tampered"
        ) from e

    return plaintext

#AEGIS-256
def AEGIS256_Encrypt(
    plaintext: bytes,
    associated_data: bytes,
    nonce: bytes,
    key: bytes
) -> bytes:
    """
    Encrypt using AEGIS-256 (bytes-only).

    Args:
        plaintext (bytes): Plaintext.
        associated_data (bytes): Associated data (AD).
        nonce (bytes): Nonce (IV).
        key (bytes): Encryption key.

    Returns:
        bytes: Ciphertext (including authentication tag).
    """

    return pysodium.crypto_aead_aegis256_encrypt(
        plaintext,
        associated_data,
        nonce,
        key
    )

def AEGIS256_Decrypt(
    ciphertext: bytes,
    associated_data: bytes,
    nonce: bytes,
    key: bytes
) -> bytes:

    try:
        return pysodium.crypto_aead_aegis256_decrypt(
            ciphertext,
            associated_data,
            nonce,
            key
        )
    except ValueError as e:
        raise ValueError(
            "Decryption failed: authentication tag invalid or data tampered"
        ) from e

#ASCON128
def ASCON_Encrypt(IV: bytes, AD: bytes, KEY: bytes, PT: bytes) -> bytes:


    CTTAG = ascon.encrypt(KEY, IV, AD, PT, variant="Ascon-128a")
    return CTTAG


def ASCON_Decrypt(IV: bytes, AD: bytes, KEY: bytes, CTTAG: bytes) -> bytes:

    PT = ascon.decrypt(KEY, IV, AD, CTTAG, variant="Ascon-128a")

    if PT is None:
        raise ValueError("标签验证失败 / Authentication tag verification failed")

    return PT

#AES256
def AES_Enc(plaintext: bytes, key: bytes) -> bytes:

    nonce = get_random_bytes(12)  # recommended length for GCM
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce + ciphertext + tag


def AES_Dec(combined: bytes, key: bytes) -> bytes:

    nonce = combined[:12]
    tag = combined[-16:]
    ciphertext = combined[12:-16]

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        return cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError as e:
        raise ValueError(
            "GCM decryption failed: authentication tag mismatch (tampered or wrong key/nonce)"
        ) from e

#GIFT-COFB
CRYPTO_KEYBYTES = 16
CRYPTO_NPUBBYTES = 16
CRYPTO_ABYTES = 16
TAGBYTES = CRYPTO_ABYTES

# GIFT-128 S-box 轮常数
GIFT_RC = [
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
    0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
    0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
    0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A
]


# =============================================================================
# GIFT-128 分组密码实现 (Block Cipher)
# 对应 gift128.c
# =============================================================================

def rowperm(S, B0_pos, B1_pos, B2_pos, B3_pos):
    T = 0
    for b in range(8):
        T |= ((S >> (4 * b + 0)) & 0x1) << (b + 8 * B0_pos)
        T |= ((S >> (4 * b + 1)) & 0x1) << (b + 8 * B1_pos)
        T |= ((S >> (4 * b + 2)) & 0x1) << (b + 8 * B2_pos)
        T |= ((S >> (4 * b + 3)) & 0x1) << (b + 8 * B3_pos)
    return T


def giftb128(P, K):
    """
    执行 GIFT-128 加密块。
    P: 16字节输入
    K: 16字节密钥
    返回: 16字节密文 (bytearray)
    """
    # 将输入字节解包为4个32位整数 (Big Endian)
    S = list(struct.unpack(">4I", P))
    W = list(struct.unpack(">8H", K))

    for round_idx in range(40):
        # === SubCells ===
        S[1] ^= S[0] & S[2]
        S[0] ^= S[1] & S[3]
        S[2] ^= S[0] | S[1]
        S[3] ^= S[2]
        S[1] ^= S[3]
        S[3] ^= 0xffffffff
        S[2] ^= S[0] & S[1]

        # Swap
        S[0], S[3] = S[3], S[0]

        # === PermBits ===
        S[0] = rowperm(S[0], 0, 3, 2, 1)
        S[1] = rowperm(S[1], 1, 0, 3, 2)
        S[2] = rowperm(S[2], 2, 1, 0, 3)
        S[3] = rowperm(S[3], 3, 2, 1, 0)

        # === AddRoundKey ===
        S[2] ^= ((W[2] << 16) | W[3])
        S[1] ^= ((W[6] << 16) | W[7])

        # Add round constant
        S[3] ^= 0x80000000 ^ GIFT_RC[round_idx]

        # === Key state update ===
        T6 = (W[6] >> 2) | ((W[6] << 14) & 0xFFFF)
        T7 = (W[7] >> 12) | ((W[7] << 4) & 0xFFFF)

        W[7] = W[5];
        W[6] = W[4];
        W[5] = W[3];
        W[4] = W[2]
        W[3] = W[1];
        W[2] = W[0];
        W[1] = T7;
        W[0] = T6

    return bytearray(struct.pack(">4I", *S))


# =============================================================================
# COFB 模式辅助函数 (Helper Functions)
# 对应 encrypt.c
# =============================================================================

def padding(s, no_of_bytes):
    tmp = bytearray(16)
    if no_of_bytes == 0:
        tmp[0] = 0x80
    elif no_of_bytes < 16:
        tmp[:no_of_bytes] = s[:no_of_bytes]
        tmp[no_of_bytes] = 0x80
    else:
        tmp[:] = s[:16]
    return tmp


def xor_block(s1, s2, no_of_bytes):
    d = bytearray(len(s1))
    d[:] = s1[:]
    for i in range(no_of_bytes):
        d[i] = s1[i] ^ s2[i]
    return d


def xor_topbar_block(s1, s2):
    d = bytearray(s1)
    for i in range(8):
        d[i] ^= s2[i]
    return d


def double_half_block(s):
    tmp = bytearray(8)
    for i in range(7):
        tmp[i] = ((s[i] << 1) & 0xFF) | (s[i + 1] >> 7)
    tmp[7] = ((s[7] << 1) & 0xFF) ^ ((s[0] >> 7) * 27)
    return tmp


def triple_half_block(s):
    d_doubled = double_half_block(s)
    d = bytearray(8)
    for i in range(8):
        d[i] = s[i] ^ d_doubled[i]
    return d


def G(s):
    d = bytearray(16)
    for i in range(8):
        d[i] = s[8 + i]
    for i in range(7):
        d[i + 8] = ((s[i] << 1) & 0xFF) | (s[i + 1] >> 7)
    d[7 + 8] = ((s[7] << 1) & 0xFF) | (s[0] >> 7)
    return d


def pho1(Y, M, no_of_bytes):
    new_Y = G(Y)
    tmpM = padding(M, no_of_bytes)
    d = xor_block(new_Y, tmpM, 16)
    return d, new_Y


def pho(Y, M, X_input, no_of_bytes):
    C = xor_block(Y, M, no_of_bytes)
    X, new_Y = pho1(Y, M, no_of_bytes)
    return C, X, new_Y


def phoprime(Y, C, X_input, no_of_bytes):
    M = xor_block(Y, C, no_of_bytes)
    X, new_Y = pho1(Y, M, no_of_bytes)
    return M, X, new_Y

def _cofb_crypt(k, n, a, data, encrypting=True):
    """
    内部使用的核心 COFB 处理函数。
    """
    k = bytearray(k)
    n = bytearray(n)
    a = bytearray(a)
    data = bytearray(data)

    if not encrypting:
        if len(data) < TAGBYTES:
            return None
        input_data = data[:-TAGBYTES]
        expected_tag = data[-TAGBYTES:]
    else:
        input_data = data

    alen = len(a)
    inlen = len(input_data)

    emptyA = (alen == 0)
    emptyM = (inlen == 0)

    out_buf = bytearray()

    # === Mask-Gen ===
    input_block = bytearray(n)
    Y = giftb128(input_block, k)
    offset = Y[:8]

    # === Process AD ===
    ad_idx = 0
    while alen > 16:
        block_a = a[ad_idx: ad_idx + 16]
        X_partial, Y = pho1(Y, block_a, 16)
        offset = double_half_block(offset)
        input_block = xor_topbar_block(X_partial, offset)
        Y = giftb128(input_block, k)
        ad_idx += 16
        alen -= 16

    offset = triple_half_block(offset)
    if (alen % 16 != 0) or emptyA:
        offset = triple_half_block(offset)

    if emptyM:
        offset = triple_half_block(offset)
        offset = triple_half_block(offset)

    block_a = a[ad_idx: ad_idx + alen]
    X_partial, Y = pho1(Y, block_a, alen)
    input_block = xor_topbar_block(X_partial, offset)
    Y = giftb128(input_block, k)

    # === Process Message ===
    data_idx = 0
    while inlen > 16:
        offset = double_half_block(offset)
        block_m = input_data[data_idx: data_idx + 16]

        if encrypting:
            block_out, X_partial, Y = pho(Y, block_m, input_block, 16)
        else:
            block_out, X_partial, Y = phoprime(Y, block_m, input_block, 16)

        out_buf.extend(block_out[:16])
        input_block = xor_topbar_block(X_partial, offset)
        Y = giftb128(input_block, k)
        data_idx += 16
        inlen -= 16

    if not emptyM:
        offset = triple_half_block(offset)
        if inlen % 16 != 0:
            offset = triple_half_block(offset)

        block_m = input_data[data_idx: data_idx + inlen]

        if encrypting:
            block_out, X_partial, Y = pho(Y, block_m, input_block, inlen)
            out_buf.extend(block_out[:inlen])
        else:
            block_out, X_partial, Y = phoprime(Y, block_m, input_block, inlen)
            out_buf.extend(block_out[:inlen])

        input_block = xor_topbar_block(X_partial, offset)
        Y = giftb128(input_block, k)

    # Tag Generation
    Tag = Y

    if encrypting:
        return bytes(out_buf + Tag)  # 返回 bytes
    else:
        if Tag == expected_tag:
            return bytes(out_buf)  # 返回 bytes
        else:
            return None


def GIFT_enc(nonce, associated_data,key,  plaintext):
    """
    GIFT-COFB 加密函数。

    参数:
        key (bytes): 16字节密钥
        nonce (bytes): 16字节随机数
        associated_data (bytes): 关联数据
        plaintext (bytes): 明文

    返回:
        bytes: 密文 + Tag (Ciphertext || Tag)
    """
    if len(key) != 16 or len(nonce) != 16:
        raise ValueError("Key and Nonce must be 16 bytes")

    return _cofb_crypt(key, nonce, associated_data, plaintext, encrypting=True)


def GIFT_dec(nonce, associated_data, key, ciphertext):
    """
    GIFT-COFB 解密函数。

    参数:
        key (bytes): 16字节密钥
        nonce (bytes): 16字节随机数
        associated_data (bytes): 关联数据
        ciphertext (bytes): 密文 (包含末尾的 Tag)

    返回:
        bytes: 解密后的明文
        None: 如果 Tag 验证失败
    """
    if len(key) != 16 or len(nonce) != 16:
        raise ValueError("Key and Nonce must be 16 bytes")

    return _cofb_crypt(key, nonce, associated_data, ciphertext, encrypting=False)


from ecpy.curves import Curve
import random

curve = Curve.get_curve('secp256k1')
SCALAR_SIZE = 32

def Get_base_point():
    """
    获取secp256k1的基点（压缩格式）

    返回:
        bytes: 压缩后的基点（33字节）
    """

    curve = Curve.get_curve('secp256k1')

    G = curve.generator

    compressed_list = curve.encode_point(G, compressed=True)


    compressed_bytes = bytes(compressed_list)

    return compressed_bytes

def decompress_secp256k1_point(compressed_point):
    """
    解压缩secp256k1点

    参数:
        compressed_point (bytes): 压缩格式的点（33字节）

    返回:
        Point: 完整的点对象
    """

    curve = Curve.get_curve('secp256k1')


    compressed_list = list(compressed_point)

    point = curve.decode_point(compressed_list)

    return point


def get_curve_order():
    """
    获取secp256k1曲线的阶（有限域的大小）

    返回:
        int: 曲线的阶
    """
    return curve.order


def get_scalar_size():
    """
    获取标量的标准字节长度

    返回:
        int: 标量的字节长度（32字节）
    """
    return SCALAR_SIZE


def Generate_scalar():
    """
    生成一个有效的标量（私钥），返回字节格式

    返回:
        bytes: 32字节的标量
    """
    n = curve.order

    while True:
        scalar_bytes = random.randbytes(SCALAR_SIZE)
        scalar_int = int.from_bytes(scalar_bytes, 'big')

        if 1 <= scalar_int < n:
            return scalar_bytes


def int_to_scalar_bytes(k_int):
    """
    将整数转换为标量字节格式

    参数:
        k_int (int): 整数标量

    返回:
        bytes: 32字节的标量
    """
    if not isinstance(k_int, int):
        raise TypeError("输入必须是整数")

    n = curve.order
    if not (1 <= k_int < n):
        raise ValueError(f"标量必须在[1, {n - 1}]范围内")

    return k_int.to_bytes(SCALAR_SIZE, 'big')


def scalar_bytes_to_int(k_bytes):
    """
    将标量字节格式转换为整数

    参数:
        k_bytes (bytes): 32字节的标量

    返回:
        int: 整数标量
    """
    if not isinstance(k_bytes, bytes):
        raise TypeError("输入必须是字节格式")

    if len(k_bytes) != SCALAR_SIZE:
        raise ValueError(f"标量必须是{SCALAR_SIZE}字节")

    k_int = int.from_bytes(k_bytes, 'big')
    n = curve.order

    if not (1 <= k_int < n):
        raise ValueError(f"标量必须在[1, {n - 1}]范围内")

    return k_int

def Scalar_add(k1_bytes, k2_bytes):
    """
    标量加法（模曲线阶），输入输出都是字节格式

    参数:
        k1_bytes (bytes): 第一个标量（32字节）
        k2_bytes (bytes): 第二个标量（32字节）

    返回:
        bytes: (k1 + k2) mod n 的32字节表示
    """
    # 转换为整数
    k1 = scalar_bytes_to_int(k1_bytes)
    k2 = scalar_bytes_to_int(k2_bytes)

    n = curve.order
    result = (k1 + k2) % n

    if result == 0:
        result = 1

    return int_to_scalar_bytes(result)


def Scalar_sub(k1_bytes, k2_bytes):
    """
    标量减法（模曲线阶），输入输出都是字节格式

    参数:
        k1_bytes (bytes): 第一个标量（32字节）
        k2_bytes (bytes): 第二个标量（32字节）

    返回:
        bytes: (k1 - k2) mod n 的32字节表示
    """

    k1 = scalar_bytes_to_int(k1_bytes)
    k2 = scalar_bytes_to_int(k2_bytes)


    n = curve.order
    result = (k1 - k2) % n


    if result == 0:
        result = 1


    return int_to_scalar_bytes(result)


def Scalar_mul(k1_bytes, k2_bytes):
    """
    标量乘法（模曲线阶），输入输出都是字节格式

    参数:
        k1_bytes (bytes): 第一个标量（32字节）
        k2_bytes (bytes): 第二个标量（32字节）

    返回:
        bytes: (k1 * k2) mod n 的32字节表示
    """

    k1 = scalar_bytes_to_int(k1_bytes)
    k2 = scalar_bytes_to_int(k2_bytes)


    n = curve.order
    result = (k1 * k2) % n

    if result == 0:
        result = 1

    return int_to_scalar_bytes(result)


def Scalar_inverse(k_bytes):
    """
    计算标量的模逆，输入输出都是字节格式

    参数:
        k_bytes (bytes): 标量（32字节）

    返回:
        bytes: k的模逆的32字节表示
    """

    k = scalar_bytes_to_int(k_bytes)
    n = curve.order

    def extended_gcd(a, b):
        if b == 0:
            return (1, 0, a)
        else:
            x, y, gcd = extended_gcd(b, a % b)
            return (y, x - (a // b) * y, gcd)

    x, _, gcd = extended_gcd(k, n)

    if gcd != 1:
        raise ValueError("标量没有模逆（与曲线阶不互质）")

    result = x % n
    if result == 0:
        result = 1

    return int_to_scalar_bytes(result)


def Scalar_neg(k_bytes):
    """
    计算标量的负元（加法逆元），输入输出都是字节格式

    参数:
        k_bytes (bytes): 标量（32字节）

    返回:
        bytes: -k mod n 的32字节表示
    """
    k = scalar_bytes_to_int(k_bytes)
    n = curve.order

    result = (-k) % n

    if result == 0:
        result = 1

    return int_to_scalar_bytes(result)


def Scalar_pow(k_bytes, exponent):
    """
    标量的幂运算（模曲线阶）

    参数:
        k_bytes (bytes): 标量（32字节）
        exponent (int): 指数

    返回:
        bytes: k^exponent mod n 的32字节表示
    """
    k = scalar_bytes_to_int(k_bytes)
    n = curve.order

    result = pow(k, exponent, n)


    if result == 0:
        result = 1

    return int_to_scalar_bytes(result)


def validate_scalar_bytes(k_bytes):
    """
    验证标量字节是否有效

    参数:
        k_bytes (bytes): 标量的字节表示

    返回:
        bool: 如果标量有效返回True，否则返回False
    """
    try:
        scalar_bytes_to_int(k_bytes)
        return True
    except (ValueError, TypeError):
        return False


def scalar_from_int_safe(k_int):
    """
    安全地从整数创建标量字节，自动进行模约减

    参数:
        k_int (int): 整数

    返回:
        bytes: 32字节的标量
    """
    if not isinstance(k_int, int):
        raise TypeError("输入必须是整数")

    n = curve.order
    # 模约减
    k_mod = k_int % n

    # 确保不为0
    if k_mod == 0:
        k_mod = 1

    return int_to_scalar_bytes(k_mod)


def Scalar_add_int(k_bytes, k_int):
    """
    标量字节与整数相加

    参数:
        k_bytes (bytes): 标量字节
        k_int (int): 整数

    返回:
        bytes: 结果的32字节表示
    """
    # 将标量字节转换为整数
    k1 = scalar_bytes_to_int(k_bytes)

    # 执行模加法
    n = curve.order
    result = (k1 + k_int) % n

    # 确保结果不为0
    if result == 0:
        result = 1

    # 转换回字节
    return int_to_scalar_bytes(result)


def Scalar_mul_int(k_bytes, k_int):
    """
    标量字节与整数相乘

    参数:
        k_bytes (bytes): 标量字节
        k_int (int): 整数

    返回:
        bytes: 结果的32字节表示
    """
    k1 = scalar_bytes_to_int(k_bytes)

    n = curve.order
    result = (k1 * k_int) % n

    if result == 0:
        result = 1

    return int_to_scalar_bytes(result)

def ECC_Point_scalar_multiply(compressed_point, scalar_bytes):
    """
    点乘运算：标量乘以点

    参数:
        compressed_point (bytes): 压缩格式的点（33字节）
        scalar_bytes (bytes): 标量（32字节）

    返回:
        bytes: 点乘结果的压缩格式（33字节）

    异常:
        ValueError: 如果输入格式无效
    """
    if not isinstance(compressed_point, bytes):
        raise TypeError("点必须是字节格式")

    if not isinstance(scalar_bytes, bytes):
        raise TypeError("标量必须是字节格式")

    if len(compressed_point) != 33:
        raise ValueError(f"压缩点应为33字节，实际为{len(compressed_point)}字节")

    if len(scalar_bytes) != SCALAR_SIZE:
        raise ValueError(f"标量应为{SCALAR_SIZE}字节，实际为{len(scalar_bytes)}字节")

    prefix = compressed_point[0]
    if prefix not in [0x02, 0x03]:
        raise ValueError(f"无效的点前缀: {hex(prefix)}，应为0x02或0x03")

    try:
        scalar_int = scalar_bytes_to_int(scalar_bytes)
    except ValueError as e:
        raise ValueError(f"无效的标量: {e}")

    try:
        compressed_list = list(compressed_point)
        point = curve.decode_point(compressed_list)
    except Exception as e:
        raise ValueError(f"点解压缩失败: {e}")

    if not curve.is_on_curve(point):
        raise ValueError("点不在曲线上")


    try:
        result_point = scalar_int * point


        if result_point.is_infinity:
            return bytes([0x00] + [0x00] * 32)

        compressed_result_list = curve.encode_point(result_point, compressed=True)

        compressed_result = bytes(compressed_result_list)

        return compressed_result

    except Exception as e:
        raise ValueError(f"点乘运算失败: {e}")


def ECC_Points_add(*compressed_points):
    """
    多个点的加法运算（可变参数版本）

    参数:
        *compressed_points: 可变数量的压缩格式的点（每个点33字节）

    返回:
        bytes: 点加结果的压缩格式（33字节）

    异常:
        ValueError: 如果输入格式无效或点不在曲线上
    """
    points_list = list(compressed_points)

    if len(points_list) == 0:
        raise ValueError("至少需要提供一个点")

    if len(points_list) == 1:
        single_point = points_list[0]
        if not isinstance(single_point, bytes) or len(single_point) != 33:
            raise ValueError("点必须是33字节的压缩格式")
        return single_point

    try:
        first_point_list = list(points_list[0])
        result_point = curve.decode_point(first_point_list)

        if not curve.is_on_curve(result_point):
            raise ValueError("第一个点不在曲线上")

    except Exception as e:
        raise ValueError(f"第一个点解压缩失败: {e}")

    for i in range(1, len(points_list)):
        compressed_point = points_list[i]

        if not isinstance(compressed_point, bytes) or len(compressed_point) != 33:
            raise ValueError(f"第{i + 1}个点必须是33字节的压缩格式")

        try:
            point_list = list(compressed_point)
            current_point = curve.decode_point(point_list)

            if not curve.is_on_curve(current_point):
                raise ValueError(f"第{i + 1}个点不在曲线上")

        except Exception as e:
            raise ValueError(f"第{i + 1}个点解压缩失败: {e}")

        try:
            result_point = result_point + current_point

            if result_point.is_infinity:
                continue

        except Exception as e:
            raise ValueError(f"点加法运算失败: {e}")

    if result_point.is_infinity:
        # 返回无穷远点的表示（全零字节）
        return bytes([0x00] + [0x00] * 32)

    try:
        compressed_result_list = curve.encode_point(result_point, compressed=True)
        compressed_result = bytes(compressed_result_list)
        return compressed_result
    except Exception as e:
        raise ValueError(f"结果点压缩失败: {e}")

def ECC_Points_sub(first_compressed_point, *other_compressed_points):
    """
    多个点的减法运算：first - p2 - p3 - ...（可变参数版本）

    参数:
        first_compressed_point (bytes): 被减数点（33字节压缩格式）
        *other_compressed_points: 可变数量的减数点（每个点33字节压缩格式）

    返回:
        bytes: 点减结果的压缩格式（33字节）

    异常:
        ValueError: 如果输入格式无效或点不在曲线上
    """
    if not isinstance(first_compressed_point, bytes) or len(first_compressed_point) != 33:
        raise ValueError("第一个点必须是33字节的压缩格式")

    try:
        first_point = curve.decode_point(list(first_compressed_point))
        if not curve.is_on_curve(first_point):
            raise ValueError("第一个点不在曲线上")
    except Exception as e:
        raise ValueError(f"第一个点解压缩失败: {e}")

    if len(other_compressed_points) == 0:
        return first_compressed_point

    result_point = first_point

    for i, compressed_point in enumerate(other_compressed_points, start=2):
        if not isinstance(compressed_point, bytes) or len(compressed_point) != 33:
            raise ValueError(f"第{i}个点必须是33字节的压缩格式")

        try:
            q = curve.decode_point(list(compressed_point))
            if not curve.is_on_curve(q):
                raise ValueError(f"第{i}个点不在曲线上")
        except Exception as e:
            raise ValueError(f"第{i}个点解压缩失败: {e}")

        try:
            q_neg = -q
            result_point = result_point + q_neg
        except Exception as e:
            raise ValueError(f"点减法运算失败: {e}")

    if result_point.is_infinity:
        return bytes([0x00] + [0x00] * 32)

    try:
        return bytes(curve.encode_point(result_point, compressed=True))
    except Exception as e:
        raise ValueError(f"结果点压缩失败: {e}")

###################################################################################
def Kyber_key_generation():

    PK, SK = Kyber512.keygen()

    return SK,PK


def Kyber_Encaps(PK):

    Shared_key, Ciphertext = Kyber512.encaps(PK)

    return Shared_key, Ciphertext


def Kyber_Decaps(SK,Ciphertext):

    K= Kyber512.decaps(SK, Ciphertext)

    return K

###################################################################################
# hmac
def Hmac(data: bytes) -> bytes:
    """
    Input:  data (bytes)
    Output: HMAC-SHA256 tag (bytes, 32 bytes)
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    return hmac.new(Hash(data), bytes(data), hashlib.sha256).digest()


###################################################################################
# 切比雪夫混沌映射
def Chebyshev_chaotic_map(n_bytes, x_bytes):
    """
    快速幂整数 Chebyshev
    n_bytes: bytes 映射阶数，以字节格式表示
    x_bytes: bytes 初值，以字节格式表示
    返回: T_n(x) 的 32 字节输出 (bytes 格式)
    """
    q = 89083468397509762891851206247005308137861557330794710300215673333330753300997  # 固定模数

    n = int.from_bytes(n_bytes, 'big')
    x = int.from_bytes(x_bytes, 'big')

    if n == 0:
        return (1).to_bytes(32, 'big')
    if n == 1:
        return (x % q).to_bytes(32, 'big')

    def mat_mult(A, B):
        return [
            [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % q, (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % q],
            [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % q, (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % q]
        ]

    def mat_pow(M, exp):
        result = [[1, 0], [0, 1]]  # 单位矩阵
        while exp > 0:
            if exp % 2 == 1:
                result = mat_mult(result, M)
            M = mat_mult(M, M)
            exp //= 2
        return result

    M = [[2 * x % q, q - 1], [1, 0]]  # 注意 -1 mod q = q-1
    Mp = mat_pow(M, n - 1)
    T_n = (Mp[0][0] * x + Mp[0][1] * 1) % q

    return T_n.to_bytes(32, 'big')

