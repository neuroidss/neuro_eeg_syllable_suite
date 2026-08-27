#!/usr/bin/env python3
"""
🧠 NEUROCANVAS v35.0: REAL-TIME THETA & SYLLABLE RATE TELEMETRY
- ПОЛНАЯ ТЕЛЕМЕТРИЯ СКОРОСТИ:
    * Точная частота Теты (Hz) и длительность слога (мс).
    * Скорость речи: слогов/сек и слогов/мин в реальном времени.
    * Эквивалент музыкального темпа (Psytrance BPM).
    * Длительность одного кванта рабочей памяти (Gamma slot dt).
- THETA-SLAVED АВТО-ТЕСТ: Слоги переключаются строго в такт биологической Тете.
- 100% ЧЕСТНЫЙ ЧАТ: Только реальный вывод нейродекодера.
- ОРТОГОНАЛЬНАЯ АКУСТИКА СТИВЕНСА-КЛАТТА + ДИФТОНГИ [Я, Ю].
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import math
import numpy as np
import pygame
import sounddevice as sd
import queue
import threading
import torch

from neuro_heterarchy_core import (
    HeterarchicalBrainEngine, NUM_CHANNELS, NUM_FREQS, NUM_PAIRS,
    COORDS_X, COORDS_Y, I_IDX, J_IDX
)

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WIDTH, HEIGHT = 1280, 800
CENTER_X, CENTER_Y = 380, 390
R_MAJOR, R_MINOR = 220.0, 90.0

SRC_X, SRC_Y = COORDS_X[I_IDX] / 13.0, COORDS_Y[I_IDX] / 13.0
DST_X, DST_Y = COORDS_X[J_IDX] / 13.0, COORDS_Y[J_IDX] / 13.0
DX_PAIRS = (DST_X - SRC_X).astype(np.float32)
DY_PAIRS = (DST_Y - SRC_Y).astype(np.float32)
MID_X_PAIRS = ((SRC_X + DST_X) * 0.5).astype(np.float32)
MID_Y_PAIRS = ((SRC_Y + DST_Y) * 0.5).astype(np.float32)

CURL_WEIGHTS = (MID_X_PAIRS * DY_PAIRS - MID_Y_PAIRS * DX_PAIRS).astype(np.float32)
DIV_WEIGHTS = (MID_X_PAIRS * DX_PAIRS + MID_Y_PAIRS * DY_PAIRS).astype(np.float32)

RADII = np.hypot(COORDS_X, COORDS_Y)
IS_INNER = RADII < 8.0

idx_inner_inner, idx_outer_outer, idx_inner_outer = [], [], []
for p in range(NUM_PAIRS):
    ch_i, ch_j = I_IDX[p], J_IDX[p]
    if IS_INNER[ch_i] and IS_INNER[ch_j]: idx_inner_inner.append(p)
    elif not IS_INNER[ch_i] and not IS_INNER[ch_j]: idx_outer_outer.append(p)
    else: idx_inner_outer.append(p)

IDX_CORE_SET = set(idx_inner_inner)
IDX_OUTER_SET = set(idx_outer_outer)
IDX_CROSS_SET = set(idx_inner_outer)

DX_GPU = torch.from_numpy(DX_PAIRS).to(DEVICE)
DY_GPU = torch.from_numpy(DY_PAIRS).to(DEVICE)
MID_X_GPU = torch.from_numpy(MID_X_PAIRS).to(DEVICE)
MID_Y_GPU = torch.from_numpy(MID_Y_PAIRS).to(DEVICE)
CURL_GPU = torch.from_numpy(CURL_WEIGHTS).to(DEVICE)
DIV_GPU = torch.from_numpy(DIV_WEIGHTS).to(DEVICE)
IDX_CORE = torch.tensor(idx_inner_inner, device=DEVICE, dtype=torch.long)
IDX_OUTER = torch.tensor(idx_outer_outer, device=DEVICE, dtype=torch.long)
IDX_CROSS = torch.tensor(idx_inner_outer, device=DEVICE, dtype=torch.long)

def torus_to_3d(theta, phi):
    return (R_MAJOR + R_MINOR * math.cos(phi)) * math.cos(theta), (R_MAJOR + R_MINOR * math.cos(phi)) * math.sin(theta), R_MINOR * math.sin(phi)

def project_3d(x, y, z, pitch=0.85, yaw=0.0):
    x1, y1 = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
    y2, z2 = y1 * math.cos(pitch) - z * math.sin(pitch), y1 * math.sin(pitch) + z * math.cos(pitch)
    return int(CENTER_X + x1), int(CENTER_Y - y2), z2

# ==============================================================================
# 🧬 ЧИСТЫЙ ОРТОГОНАЛЬНЫЙ ГЕНЕРАТОР КОГЕРЕНТНОСТИ (Mode 1)
# ==============================================================================
def generate_syllable_pac_tensor(c_info, v_info):
    v_name, v_f1, v_f2, v_f3, v_th, v_ph = v_info[:6]
    
    k_vx, k_vy = math.cos(v_th), math.sin(v_th)
    phase_diff_vowel = (DX_PAIRS * k_vx + DY_PAIRS * k_vy) * 0.40
    
    is_iotated = v_name in ("Я", "Ю")
    if is_iotated:
        phase_diff_iotated_on = (DX_PAIRS * 1.0 + DY_PAIRS * 0.0) * 0.40
    
    if c_info is not None:
        c_th = c_info['th']
        c_type = c_info['type']
        k_cx, k_cy = math.cos(c_th), math.sin(c_th)
        phase_diff_cons = (DX_PAIRS * k_cx + DY_PAIRS * k_cy) * 0.40
        is_voiced = c_info.get('voice_bar', False)
    else:
        c_th = v_th
        c_type = 'VOWEL'
        phase_diff_cons = phase_diff_vowel
        is_voiced = False

    pac_matrix = np.zeros((NUM_FREQS, NUM_PAIRS), dtype=np.float32)
    rx_vec = np.zeros(NUM_FREQS, dtype=np.float32)
    ry_vec = np.zeros(NUM_FREQS, dtype=np.float32)

    for k in range(NUM_FREQS):
        prog = k / 31.0
        
        if c_type == 'VOWEL':
            if is_iotated and prog < 0.40:
                blend = prog / 0.40
                p_diff = (1.0 - blend) * phase_diff_iotated_on + blend * phase_diff_vowel
            else:
                p_diff = phase_diff_vowel
            pac_matrix[k, :] = np.sin(p_diff)
            ry_val = math.cos(v_ph) * -0.9
            rx_val = 0.0

        else:
            if prog < 0.20:
                base_p = phase_diff_cons
                pac_matrix[k, :] = np.sin(base_p)
                if c_type == 'STOP':
                    pac_matrix[k, idx_inner_outer] -= 0.85
                elif c_type == 'FRICATIVE':
                    pac_matrix[k, idx_outer_outer] += 0.85
                    
                if is_voiced or c_type == 'NASAL':
                    pac_matrix[k, idx_inner_inner] += 0.95
                    
                ry_val = -0.8
                rx_val = 0.8 if c_type == 'FRICATIVE' else 0.0
                
            elif prog < 0.35:
                blend = (prog - 0.20) / 0.15
                target_p = phase_diff_iotated_on if is_iotated else phase_diff_vowel
                base_p = (1.0 - blend) * phase_diff_cons + blend * target_p
                pac_matrix[k, :] = np.sin(base_p)
                
                if c_type == 'STOP':
                    pac_matrix[k, idx_inner_outer] += 0.90
                elif c_type == 'FRICATIVE':
                    pac_matrix[k, idx_outer_outer] += 0.85
                    
                ry_val = 0.0
                rx_val = 0.8 if c_type == 'FRICATIVE' else 0.0
                
            else:
                if is_iotated and prog < 0.60:
                    blend = (prog - 0.35) / 0.25
                    p_diff = (1.0 - blend) * phase_diff_iotated_on + blend * phase_diff_vowel
                else:
                    p_diff = phase_diff_vowel
                pac_matrix[k, :] = np.sin(p_diff)
                ry_val = math.cos(v_ph) * -0.9
                rx_val = 0.0
                
            if c_type == 'NASAL' and prog < 0.50:
                pac_matrix[k, idx_inner_inner] += 0.95

        rx_vec[k] = rx_val
        ry_vec[k] = ry_val

    return pac_matrix, float(rx_vec.mean()), float(ry_vec.mean()), v_th, v_ph

# ==============================================================================
# ⚡ FULL BIO-PHYSICAL VOCAL TRACT ENGINE (100% CUDA)
# ==============================================================================
class StudioContinuousSpeechEngine:
    def __init__(self, sr=SAMPLE_RATE, device=DEVICE):
        self.sr, self.block_size, self.device = sr, BLOCK_SIZE, device
        self.rfft_freqs = torch.fft.rfftfreq(self.block_size, d=1.0/self.sr, device=self.device)
        self.t_vec = torch.arange(self.block_size, device=self.device, dtype=torch.float32)
        
        self.theta_phase = 0.0
        self.glottal_phase = 0.0
        self.jitter_phase = 0.0
        
        self.cur_f1, self.cur_f2, self.cur_f3 = 500.0, 1500.0, 2500.0
        self.cur_nasal = 0.0
        self.last_voicing_gain = 1.0 
        
        self.smooth_cross = 0.0
        self.smooth_outer = 0.0
        self.smooth_core = 0.0
        self.motor_plan_latched = False
        self.burst_fired = False
        self.has_detected_syllable = False
        
        self.latched_c_type = 'NONE'
        self.latched_locus_f1 = 500.0
        self.latched_locus_f2 = 1500.0
        self.latched_locus_f3 = 2500.0
        
        self.latched_burst_fc = 3000.0
        self.latched_burst_bw = 380.0
        self.latched_burst_amp = 0.0
        self.latched_fric_fc = 4500.0
        self.latched_fric_bw = 650.0
        self.latched_fric_amp = 0.0
        self.latched_is_voiced = False
        self.latched_nasal_val = 0.0

        self.gesture_mode = False
        self.g_state = 'IDLE' 
        self.g_timer = 0.0
        self.g_voice_bar = False
        self.g_burst_fc, self.g_burst_bw, self.g_burst_amp = 0.0, 0.0, 0.0
        self.g_fric_fc, self.g_fric_bw, self.g_fric_amp = 0.0, 0.0, 0.0
        self.g_tgt_f1, self.g_tgt_f2, self.g_tgt_f3 = 820.0, 1250.0, 2500.0
        self.g_pending_burst_dur = 0.0
        self.g_dur_glide = 0.035

        self.biological_theta_hz = 5.5
        self.iplv_tensor = torch.zeros((NUM_FREQS, NUM_PAIRS), device=self.device, dtype=torch.float32)
        self.avatar_th = math.pi * 0.5
        self.avatar_ph = math.pi
        self.target_rx, self.target_ry = 0.0, 0.0
        self.target_f0 = 135.0 

        self.last_emitted_eeg_syllable = ""
        self.eeg_cooldown = 0.0
        self.param_lock = threading.Lock()
        self.audio_queue = queue.Queue(maxsize=32)

    def update_state(self, iplv_32, theta_hz, rx, ry, th_avatar, ph_avatar):
        with self.param_lock:
            self.iplv_tensor.copy_(torch.from_numpy(np.abs(iplv_32)))
            self.biological_theta_hz += (theta_hz - self.biological_theta_hz) * 0.1
            self.target_rx, self.target_ry = rx, ry
            self.avatar_th = th_avatar
            self.avatar_ph = ph_avatar

    def trigger_syllable_preset(self, c_info, v_info):
        with self.param_lock:
            self.gesture_mode = True
            v_name, v_f1, v_f2, v_f3, v_th, v_ph = v_info[:6]
            v_glide_tau = v_info[6] if len(v_info) > 6 else 0.035
            
            self.g_tgt_f1, self.g_tgt_f2, self.g_tgt_f3 = v_f1, v_f2, v_f3
            self.avatar_th, self.avatar_ph = v_th, v_ph
            self.g_dur_glide = v_glide_tau
            self.g_voice_bar = False
            self.cur_nasal = 0.0
            self.g_burst_fc, self.g_fric_fc = 0.0, 0.0
            
            if v_name in ("Я", "Ю") and c_info is None:
                self.cur_f1 = 280.0
                self.cur_f2 = 2250.0
                self.cur_f3 = 2900.0
            
            if c_info is None:
                self.g_state = 'GLIDE'
                self.g_timer = 0.26
                return

            c_type = c_info['type']
            if c_type == 'STOP':
                self.cur_f1, self.cur_f2, self.cur_f3 = c_info['locus_f1'], c_info['locus_f2'], c_info['locus_f3']
                self.g_burst_fc, self.g_burst_bw, self.g_burst_amp = c_info['burst_fc'], c_info['burst_bw'], c_info['burst_amp']
                self.g_voice_bar = c_info.get('voice_bar', False)
                self.g_state = 'OCCLUSION'
                self.g_timer = c_info['dur_close']
                self.g_pending_burst_dur = c_info['dur_burst']
            elif c_type == 'FRICATIVE':
                self.cur_f1, self.cur_f2, self.cur_f3 = c_info['locus_f1'], c_info['locus_f2'], c_info['locus_f3']
                self.g_fric_fc, self.g_fric_bw, self.g_fric_amp = c_info['fric_fc'], c_info['fric_bw'], c_info['fric_amp']
                self.g_state = 'FRICATIVE'
                self.g_timer = c_info['dur_fric']
            elif c_type == 'NASAL':
                self.cur_f1, self.cur_f2, self.cur_f3 = c_info['locus_f1'], c_info['locus_f2'], c_info['locus_f3']
                self.cur_nasal = c_info['nasal_val']
                self.g_state = 'GLIDE'
                self.g_timer = 0.26

    def clear_gesture(self):
        with self.param_lock:
            self.gesture_mode = False
            self.g_state = 'IDLE'
            self.motor_plan_latched = False
            self.burst_fired = False

    def res_gain_gpu(self, fc_scalar, bw, amp):
        return amp / (1.0 + ((self.rfft_freqs - fc_scalar) / (bw * 0.5)) ** 2)

    def render_block(self):
        with self.param_lock:
            iplv = self.iplv_tensor
            live_theta_hz = max(3.0, min(8.5, self.biological_theta_hz))
            rx_val, ry_val = self.target_rx, self.target_ry
            is_gest = self.gesture_mode
            av_th, av_ph = self.avatar_th, self.avatar_ph
            block_dt = self.block_size / self.sr

        out_burst = torch.zeros(self.block_size, device=self.device)
        out_fric = torch.zeros(self.block_size, device=self.device)
        out_voice_bar = torch.zeros(self.block_size, device=self.device)
        target_voicing = 1.0
        detected_syllable_event = None

        if is_gest:
            # MODE 2: LABORATORY PRESET
            if self.g_state == 'OCCLUSION':
                self.g_timer -= block_dt
                target_voicing = 0.0
                if self.g_voice_bar:
                    vbar_fft = torch.fft.rfft(torch.randn(self.block_size, device=self.device))
                    H_bar = 1.0 / (1.0 + ((self.rfft_freqs - 150.0) / 40.0) ** 2)
                    out_voice_bar = torch.fft.irfft(vbar_fft * H_bar, n=self.block_size) * 0.8
                if self.g_timer <= 0:
                    self.g_state = 'BURST'
                    self.g_timer = self.g_pending_burst_dur

            elif self.g_state == 'BURST':
                self.g_timer -= block_dt
                target_voicing = 0.2
                raw_b = torch.randn(self.block_size, device=self.device) * torch.exp(-torch.linspace(0, 5.0, self.block_size, device=self.device))
                H_burst = 1.0 / (1.0 + ((self.rfft_freqs - self.g_burst_fc) / (self.g_burst_bw * 0.5)) ** 4)
                out_burst = torch.fft.irfft(torch.fft.rfft(raw_b) * H_burst, n=self.block_size) * self.g_burst_amp
                if self.g_timer <= 0:
                    self.g_state = 'GLIDE'
                    self.g_timer = 0.24

            elif self.g_state == 'FRICATIVE':
                self.g_timer -= block_dt
                target_voicing = 0.1
                raw_f = torch.randn(self.block_size, device=self.device)
                H_fric = 1.0 / (1.0 + ((self.rfft_freqs - self.g_fric_fc) / (self.g_fric_bw * 0.5)) ** 4)
                out_fric = torch.fft.irfft(torch.fft.rfft(raw_f) * H_fric, n=self.block_size) * self.g_fric_amp
                if self.g_timer <= 0:
                    self.g_state = 'GLIDE'
                    self.g_timer = 0.24

            if self.g_state in ('GLIDE', 'VOWEL'):
                self.g_timer -= block_dt
                target_voicing = 1.0
                glide_alpha = min(1.0, block_dt / max(0.015, self.g_dur_glide))
                self.cur_f1 += (self.g_tgt_f1 - self.cur_f1) * glide_alpha
                self.cur_f2 += (self.g_tgt_f2 - self.cur_f2) * glide_alpha
                self.cur_f3 += (self.g_tgt_f3 - self.cur_f3) * glide_alpha
                self.cur_nasal *= 0.80

        else:
            # MODE 0 & 1: STRICT WORKING MEMORY 2.0 PACEMAKER
            with torch.inference_mode():
                theta_inc = 2.0 * math.pi * live_theta_hz / self.sr
                t_phase_accum = self.theta_phase + self.t_vec * theta_inc
                self.theta_phase = (t_phase_accum[-1].item() + theta_inc) % (2.0 * math.pi)

                phase_array = (t_phase_accum / (2.0 * math.pi)) % 1.0
                warp_factor = 2.0 ** -ry_val 
                warped_array = phase_array ** warp_factor 
                phase_val = warped_array.mean().item()
                
                float_idx = warped_array * 32.0 
                idx_0 = float_idx.long() % 32
                idx_1 = (idx_0 + 1) % 32
                alpha = (float_idx - float_idx.floor()).unsqueeze(1)
                tensor_stream = iplv[idx_0, :] * (1.0 - alpha) + iplv[idx_1, :] * alpha

                Vx = torch.sum(tensor_stream * DX_GPU, dim=1)
                Vy = torch.sum(tensor_stream * DY_GPU, dim=1)
                th_flow = (torch.atan2(Vy, Vx) % (2.0 * math.pi)).mean().item()
                th_deg = math.degrees(th_flow) % 360

                cross_act = torch.mean(tensor_stream[:, IDX_CROSS], dim=1).mean().item()
                outer_act = torch.mean(tensor_stream[:, IDX_OUTER], dim=1).mean().item()
                core_act  = torch.mean(tensor_stream[:, IDX_CORE],  dim=1).mean().item()

                self.smooth_cross = self.smooth_cross * 0.75 + cross_act * 0.25
                self.smooth_outer = self.smooth_outer * 0.75 + outer_act * 0.25
                self.smooth_core  = self.smooth_core  * 0.75 + core_act  * 0.25

                if phase_val > 0.90:
                    self.motor_plan_latched = False
                    self.burst_fired = False

                # ==================================================================
                # 2. ОРТОГОНАЛЬНЫЙ КЛАССИФИКАТОР ЛОКУСОВ
                # ==================================================================
                if phase_val < 0.15 and not self.motor_plan_latched:
                    self.motor_plan_latched = True
                    self.has_detected_syllable = False 
                    
                    is_stop = (self.smooth_cross > 0.35) or (self.smooth_cross < -0.35)
                    is_fric = (self.smooth_outer > 0.45) and not is_stop
                    is_nasal = (self.smooth_core > 0.40) and not is_fric and not is_stop

                    if is_stop:
                        self.latched_c_type = 'STOP'
                        self.latched_is_voiced = self.smooth_core > 0.25
                        
                        if 315 <= th_deg or th_deg < 45: # [Т / Д]
                            self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 180.0, 1800.0, 2800.0
                            self.latched_burst_fc, self.latched_burst_bw, self.latched_burst_amp = (3800.0 if self.latched_is_voiced else 4200.0), 400.0, (2.8 if self.latched_is_voiced else 3.8)
                        elif 135 <= th_deg < 225: # [К / Г] Velar Pinch
                            self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 200.0, 2100.0, 2300.0
                            self.latched_burst_fc, self.latched_burst_bw, self.latched_burst_amp = (2000.0 if self.latched_is_voiced else 2100.0), 350.0, (2.5 if self.latched_is_voiced else 3.5)
                        elif 45 <= th_deg < 135: # [Ч / Ж]
                            self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 220.0, 1900.0, 2600.0
                            self.latched_burst_fc, self.latched_burst_bw, self.latched_burst_amp = 3200.0, 400.0, 3.0
                        else: # [П / Б]
                            self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 180.0, 850.0, 2100.0
                            self.latched_burst_fc, self.latched_burst_bw, self.latched_burst_amp = (850.0 if self.latched_is_voiced else 900.0), 400.0, (2.5 if self.latched_is_voiced else 3.0)
                        self.latched_nasal_val = 0.0

                    elif is_fric:
                        self.latched_c_type = 'FRICATIVE'
                        if 315 <= th_deg or th_deg < 60: # [С] Свист 6500 Гц
                            self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 200.0, 1700.0, 2700.0
                            self.latched_fric_fc, self.latched_fric_bw, self.latched_fric_amp = 6500.0, 900.0, 3.0
                        else: # [Ш] Рев 2600 Гц
                            self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 250.0, 1900.0, 2400.0
                            self.latched_fric_fc, self.latched_fric_bw, self.latched_fric_amp = 2600.0, 600.0, 2.4
                        self.latched_nasal_val = 0.0

                    elif is_nasal:
                        self.latched_c_type = 'NASAL'
                        if 45 <= th_deg < 135: self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 250.0, 2250.0, 2800.0
                        elif 135 <= th_deg < 315: self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 250.0, 850.0, 2200.0
                        else: self.latched_locus_f1, self.latched_locus_f2, self.latched_locus_f3 = 250.0, 1600.0, 2600.0
                        self.latched_nasal_val = 0.95
                        self.latched_is_voiced = True

                    else:
                        self.latched_c_type = 'NONE'

                # ==================================================================
                # 3. АРТИКУЛЯЦИЯ И АЭРОДИНАМИКА
                # ==================================================================
                if self.latched_c_type == 'NONE':
                    target_voicing = 1.0
                    self.cur_nasal *= 0.8
                    norm_h = (1.0 - math.cos(av_ph)) * 0.5
                    tgt_f1 = 265.0 + 560.0 * norm_h
                    tgt_f2 = 1350.0 + 750.0 * math.cos(av_th)
                    tgt_f3 = 2600.0 + 300.0 * math.cos(av_th)

                    self.cur_f1 += (tgt_f1 - self.cur_f1) * 0.35
                    self.cur_f2 += (tgt_f2 - self.cur_f2) * 0.35
                    self.cur_f3 += (tgt_f3 - self.cur_f3) * 0.35

                else:
                    if phase_val < 0.20:
                        self.cur_f1 = self.latched_locus_f1
                        self.cur_f2 = self.latched_locus_f2
                        self.cur_f3 = self.latched_locus_f3
                        self.cur_nasal = self.latched_nasal_val
                        
                        target_voicing = 0.3 if self.latched_is_voiced else 0.0
                        if self.latched_c_type == 'STOP' and self.latched_is_voiced:
                            vbar_fft = torch.fft.rfft(torch.randn(self.block_size, device=self.device))
                            H_bar = 1.0 / (1.0 + ((self.rfft_freqs - 150.0) / 40.0) ** 2)
                            out_voice_bar = torch.fft.irfft(vbar_fft * H_bar, n=self.block_size) * 0.8
                            
                    elif phase_val < 0.45:
                        target_voicing = 0.2
                        
                        if self.latched_c_type == 'STOP':
                            if not self.burst_fired and phase_val < 0.35:
                                b_mask = (warped_array >= 0.20) & (warped_array < 0.35)
                                burst_env = torch.zeros(self.block_size, device=self.device)
                                burst_env[b_mask] = torch.exp(-40.0 * (warped_array[b_mask] - 0.20))
                                
                                raw_b = torch.randn(self.block_size, device=self.device) * burst_env
                                H_burst = 1.0 / (1.0 + ((self.rfft_freqs - self.latched_burst_fc) / (self.latched_burst_bw * 0.5)) ** 4)
                                out_burst = torch.fft.irfft(torch.fft.rfft(raw_b) * H_burst, n=self.block_size) * self.latched_burst_amp
                                self.burst_fired = True 
                            
                        elif self.latched_c_type == 'FRICATIVE':
                            f_mask = (warped_array >= 0.20) & (warped_array < 0.45)
                            fric_env = torch.zeros(self.block_size, device=self.device)
                            fric_env[f_mask] = torch.sin(math.pi * (warped_array[f_mask] - 0.20) / 0.25)
                            
                            raw_f = torch.randn(self.block_size, device=self.device) * fric_env
                            H_fric = 1.0 / (1.0 + ((self.rfft_freqs - self.latched_fric_fc) / (self.latched_fric_bw * 0.5)) ** 4)
                            out_fric = torch.fft.irfft(torch.fft.rfft(raw_f) * H_fric, n=self.block_size) * self.latched_fric_amp
                            
                        elif self.latched_c_type == 'NASAL':
                            target_voicing = 1.0

                        norm_h = (1.0 - math.cos(av_ph)) * 0.5
                        tgt_f1 = 265.0 + 560.0 * norm_h
                        tgt_f2 = 1350.0 + 750.0 * math.cos(av_th)
                        tgt_f3 = 2600.0 + 300.0 * math.cos(av_th)

                        self.cur_f1 += (tgt_f1 - self.cur_f1) * 0.35
                        self.cur_f2 += (tgt_f2 - self.cur_f2) * 0.35
                        self.cur_f3 += (tgt_f3 - self.cur_f3) * 0.35
                        self.cur_nasal *= 0.8
                        
                    else:
                        target_voicing = 1.0
                        self.cur_nasal *= 0.8
                        norm_h = (1.0 - math.cos(av_ph)) * 0.5
                        tgt_f1 = 265.0 + 560.0 * norm_h
                        tgt_f2 = 1350.0 + 750.0 * math.cos(av_th)
                        tgt_f3 = 2600.0 + 300.0 * math.cos(av_th)

                        self.cur_f1 += (tgt_f1 - self.cur_f1) * 0.35
                        self.cur_f2 += (tgt_f2 - self.cur_f2) * 0.35
                        self.cur_f3 += (tgt_f3 - self.cur_f3) * 0.35

                # 4. ДЕТЕКТОР СЛОГОВ В ЧАТ (СТРОГО 1 РАЗ НА ТЕТА-ЦИКЛ)
                if phase_val > 0.45 and not self.has_detected_syllable:
                    self.has_detected_syllable = True 
                    v_candidates = [
                        ("И", 270, 2250), ("Э", 500, 1850), ("А", 820, 1250), 
                        ("О", 520, 880), ("У", 280, 720), ("Ы", 320, 1450),
                        ("Я", 820, 1450), ("Ю", 280, 850)
                    ]
                    best_v = min(v_candidates, key=lambda c: math.hypot(self.cur_f1 - c[1], (self.cur_f2 - c[2])*0.5))[0]

                    c_part = ""
                    if self.latched_c_type == 'STOP':
                        if 315 <= th_deg or th_deg < 45: c_part = "Д" if self.latched_is_voiced else "Т"
                        elif 135 <= th_deg < 225: c_part = "Г" if self.latched_is_voiced else "К"
                        elif 45 <= th_deg < 135: c_part = "Ж" if self.latched_is_voiced else "Ч"
                        else: c_part = "Б" if self.latched_is_voiced else "П"
                    elif self.latched_c_type == 'FRICATIVE':
                        c_part = "С" if (315 <= th_deg or th_deg < 60) else "Ш"
                    elif self.latched_c_type == 'NASAL':
                        if 45 <= th_deg < 135: c_part = "НЬ"
                        elif 135 <= th_deg < 315: c_part = "М"
                        else: c_part = "Н"

                    detected_syllable_event = c_part + best_v

        # ==============================================================================
        # 4. GLOTTAL EXCITATION & 5-FORMANT RESONATOR (44.1 kHz CUDA)
        # ==============================================================================
        with torch.inference_mode():
            voicing_vec = torch.linspace(self.last_voicing_gain, target_voicing, self.block_size, device=self.device)
            self.last_voicing_gain = target_voicing

            self.jitter_phase = (self.jitter_phase + 6.5 * block_dt * 2.0 * math.pi) % (2.0 * math.pi)
            f0_track = self.target_f0 * (1.0 + 0.012 * math.sin(self.jitter_phase))
            glot_phase_track = torch.cumsum(2.0 * math.pi * torch.full((self.block_size,), f0_track, device=self.device) / self.sr, dim=0) + self.glottal_phase
            self.glottal_phase = (glot_phase_track[-1].item()) % (2.0 * math.pi)
            norm_glot = (glot_phase_track / (2.0 * math.pi)) % 1.0
            
            glot_wave = torch.zeros(self.block_size, device=self.device)
            open_mask = norm_glot < 0.65
            p = norm_glot[open_mask] / 0.65
            glot_wave[open_mask] = 3.0 * (p ** 2) - 2.0 * (p ** 3)
            
            glottal_source = (torch.diff(glot_wave, prepend=glot_wave[0:1]) + (torch.rand(self.block_size, device=self.device) * 2.0 - 1.0) * 0.035) * (voicing_vec * 2.2)

            H_vocal = (self.res_gain_gpu(self.cur_f1, 80.0, 1.0) +
                       self.res_gain_gpu(self.cur_f2, 110.0, 0.85) +
                       self.res_gain_gpu(self.cur_f3, 150.0, 0.45) +
                       self.res_gain_gpu(3600.0, 220.0, 0.25) +
                       self.res_gain_gpu(4500.0, 280.0, 0.15))

            if self.cur_nasal > 0.02:
                H_nasal_pole = self.res_gain_gpu(250.0, 45.0, 2.4 * self.cur_nasal)
                H_nasal_zero = 1.0 - 0.70 * self.cur_nasal * torch.exp(-0.5 * ((self.rfft_freqs - 700.0) / 120.0) ** 2)
                H_vocal = (H_vocal * H_nasal_zero) + H_nasal_pole

            voice_out = torch.fft.irfft(torch.fft.rfft(glottal_source) * H_vocal.clamp(min=1e-5), n=self.block_size)

            mix_audio = (voice_out * 1.3) + out_voice_bar + out_burst + out_fric
            mix_audio = torch.tanh(mix_audio * 0.85) * 0.88
            out = torch.stack([mix_audio, mix_audio], dim=1)

            fft_mag = torch.abs(torch.fft.rfft(mix_audio)) 
            fft_mag_pooled = torch.nn.functional.avg_pool1d(fft_mag.unsqueeze(0).unsqueeze(0), kernel_size=4).squeeze()
            fft_log = torch.log1p(fft_mag_pooled * 15.0).cpu().numpy()

            telem = (self.cur_f1, self.cur_f2, target_voicing, live_theta_hz, av_th, av_ph, fft_log, detected_syllable_event)
            return out.cpu().numpy(), telem

def audio_thread_loop(synth, stop_event, telemetry_queue):
    def cb(outdata, frames, time_info, status):
        try:
            audio, telem = synth.audio_queue.get_nowait()
            outdata[:] = audio
            try: telemetry_queue.put_nowait(telem)
            except queue.Full: pass
        except queue.Empty:
            outdata.fill(0)
            
    for _ in range(8):
        audio, telem = synth.render_block()
        synth.audio_queue.put((audio, telem))
        
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=cb, blocksize=BLOCK_SIZE, dtype='float32'):
        while not stop_event.is_set():
            if synth.audio_queue.qsize() < 8: synth.audio_queue.put(synth.render_block())
            else: pygame.time.wait(2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas v35.0: Real-Time Theta & Syllable Rate Telemetry")
    clock = pygame.time.Clock()
    font_xs = pygame.font.SysFont("consolas", 10)
    font_sm = pygame.font.SysFont("consolas", 11, bold=True)
    font = pygame.font.SysFont("consolas", 13, bold=True)
    font_med = pygame.font.SysFont("consolas", 14, bold=True)
    font_lg = pygame.font.SysFont("consolas", 16, bold=True)
    font_chat = pygame.font.SysFont("consolas", 15, bold=True)

    engine = HeterarchicalBrainEngine()
    engine.start()

    synth = StudioContinuousSpeechEngine()
    stop_event = threading.Event()
    telemetry_queue = queue.Queue(maxsize=16)
    t_audio = threading.Thread(target=audio_thread_loop, args=(synth, stop_event, telemetry_queue))
    t_audio.start()

    operating_mode = 0
    torus_yaw = 0.0
    avatar_theta = math.pi * 0.5
    avatar_phi = math.pi

    TORUS_LANDMARKS = [
        (0.0, 0.0, "[ И / I ]", (0, 255, 255)),
        (0.0, math.pi * 0.5, "[ Э / E ]", (100, 220, 255)),
        (math.pi * 0.5, math.pi, "[ А / A ]", (255, 220, 50)),
        (math.pi, math.pi * 0.5, "[ О / O ]", (255, 140, 50)),
        (math.pi, 0.0, "[ У / U ]", (255, 80, 150)),
        (math.pi * 0.5, 0.2 * math.pi, "[ Ы / Y ]", (200, 100, 255)),
        (0.5 * math.pi, math.pi * 0.7, "[ Я / JA ]", (255, 180, 220)),
        (math.pi, 0.2 * math.pi, "[ Ю / JU ]", (255, 140, 220)),
        
        (0.0, 0.0, "[ T / D / S ]", (0, 200, 255)),
        (0.5 * math.pi, 0.0, "[ J / SH / CH ]", (150, 255, 100)),
        (math.pi, 0.0, "[ K / G / X ]", (255, 100, 50)),
        (1.5 * math.pi, 0.0, "[ P / B / M ]", (255, 50, 200))
    ]

    VOWELS = {
        pygame.K_F1: ("И", 270.0, 2250.0, 2900.0, 0.0, 0.0, 0.035),
        pygame.K_F2: ("Э", 500.0, 1850.0, 2600.0, 0.0, math.pi * 0.5, 0.035),
        pygame.K_F3: ("А", 820.0, 1250.0, 2500.0, math.pi * 0.5, math.pi, 0.035),
        pygame.K_F4: ("О", 520.0, 880.0, 2400.0, math.pi, math.pi * 0.5, 0.035),
        pygame.K_F5: ("У", 280.0, 720.0, 2250.0, math.pi, 0.0, 0.035),
        pygame.K_F6: ("Ы", 320.0, 1450.0, 2200.0, math.pi * 0.5, 0.2 * math.pi, 0.035),
        pygame.K_F7: ("Я", 820.0, 1450.0, 2600.0, math.pi * 0.5, math.pi, 0.065), 
        pygame.K_F8: ("Ю", 280.0, 850.0, 2300.0, math.pi, 0.0, 0.065),            
    }
    active_vowel_key = pygame.K_F3

    CONSONANTS = {
        pygame.K_1: {'name': 'Т', 'type': 'STOP', 'th': 0.0, 'dur_close': 0.045, 'dur_burst': 0.012, 'locus_f1': 180.0, 'locus_f2': 1800.0, 'locus_f3': 2800.0, 'burst_fc': 4200.0, 'burst_bw': 400.0, 'burst_amp': 3.8, 'voice_bar': False},
        pygame.K_2: {'name': 'К', 'type': 'STOP', 'th': math.pi, 'dur_close': 0.050, 'dur_burst': 0.016, 'locus_f1': 200.0, 'locus_f2': 2100.0, 'locus_f3': 2300.0, 'burst_fc': 2100.0, 'burst_bw': 350.0, 'burst_amp': 3.5, 'voice_bar': False},
        pygame.K_3: {'name': 'Б', 'type': 'STOP', 'th': 1.5*math.pi, 'dur_close': 0.040, 'dur_burst': 0.008, 'locus_f1': 180.0, 'locus_f2': 850.0,  'locus_f3': 2100.0, 'burst_fc': 850.0,  'burst_bw': 350.0, 'burst_amp': 2.5, 'voice_bar': True},
        pygame.K_4: {'name': 'С', 'type': 'FRICATIVE', 'th': 0.0, 'dur_fric': 0.120, 'locus_f1': 200.0, 'locus_f2': 1700.0, 'locus_f3': 2700.0, 'fric_fc': 6500.0, 'fric_bw': 900.0, 'fric_amp': 3.0},
        pygame.K_5: {'name': 'Ш', 'type': 'FRICATIVE', 'th': 0.5*math.pi, 'dur_fric': 0.110, 'locus_f1': 250.0, 'locus_f2': 1900.0, 'locus_f3': 2400.0, 'fric_fc': 2600.0, 'fric_bw': 600.0, 'fric_amp': 2.4},
        pygame.K_6: {'name': 'М', 'type': 'NASAL', 'th': 1.5*math.pi, 'locus_f1': 250.0, 'locus_f2': 850.0, 'locus_f3': 2200.0, 'nasal_val': 1.0},
        pygame.K_7: {'name': 'Д', 'type': 'STOP', 'th': 0.0, 'dur_close': 0.035, 'dur_burst': 0.010, 'locus_f1': 180.0, 'locus_f2': 1800.0, 'locus_f3': 2800.0, 'burst_fc': 3800.0, 'burst_bw': 400.0, 'burst_amp': 2.8, 'voice_bar': True},
        pygame.K_8: {'name': 'Г', 'type': 'STOP', 'th': math.pi, 'dur_close': 0.040, 'dur_burst': 0.012, 'locus_f1': 200.0, 'locus_f2': 2100.0, 'locus_f3': 2300.0, 'burst_fc': 2000.0, 'burst_bw': 350.0, 'burst_amp': 2.5, 'voice_bar': True},
        pygame.K_9: {'name': 'П', 'type': 'STOP', 'th': 1.5*math.pi, 'dur_close': 0.055, 'dur_burst': 0.010, 'locus_f1': 180.0, 'locus_f2': 850.0,  'locus_f3': 2100.0, 'burst_fc': 900.0,  'burst_bw': 400.0, 'burst_amp': 3.0, 'voice_bar': False},
        pygame.K_0: {'name': 'Н', 'type': 'NASAL', 'th': 0.0, 'locus_f1': 250.0, 'locus_f2': 1600.0, 'locus_f3': 2600.0, 'nasal_val': 0.85},
        pygame.K_MINUS: {'name': 'НЬ', 'type': 'NASAL', 'th': 0.5*math.pi, 'locus_f1': 250.0, 'locus_f2': 2250.0, 'locus_f3': 2800.0, 'nasal_val': 0.95},
    }

    active_syllable_name = "[ А ]"

    chat_syllables = []
    def add_to_chat(syll_text):
        nonlocal chat_syllables
        chat_syllables.append(syll_text)
        if len(chat_syllables) > 40:
            chat_syllables.pop(0)

    auto_test_running = False
    auto_test_c_idx = 0
    auto_test_v_idx = 0
    auto_test_latched = False
    
    c_keys_list = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0, pygame.K_MINUS]
    v_keys_list = [pygame.K_F3, pygame.K_F1, pygame.K_F5, pygame.K_F4, pygame.K_F2, pygame.K_F6, pygame.K_F7, pygame.K_F8]

    synth_target_c = None
    synth_target_v = VOWELS[pygame.K_F3]

    SPEC_W, SPEC_H = 390, 85
    spec_surface = pygame.Surface((SPEC_W, SPEC_H))
    spec_surface.fill((10, 12, 18))
    
    def map_color(val):
        val = max(0.0, min(1.0, val / 3.5))
        r = int(min(255, max(0, 255 * (val - 0.5) * 2)))
        g = int(min(255, max(0, 255 * math.sin(val * math.pi))))
        b = int(min(255, max(0, 255 * (1.0 - val * 2))))
        return (r, g, b)

    MODE_NAMES = ["[0. LIVE 16-CH EEG (BROCA)]", "[1. HONEST SYNTHETIC PAC (99% ЭТАЛОН)]", "[2. LAB PRESET MATRIX (ЭТАЛОН)]"]

    running = True
    try:
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        operating_mode = (operating_mode + 1) % 3
                        synth.clear_gesture()
                        active_syllable_name = f"РЕЖИМ: {MODE_NAMES[operating_mode]}"
                    
                    elif event.key == pygame.K_SPACE:
                        auto_test_running = True
                        auto_test_c_idx = 0
                        auto_test_v_idx = 0
                        auto_test_latched = False

                    elif event.key in VOWELS:
                        active_vowel_key = event.key
                        v_info = VOWELS[active_vowel_key]
                        synth_target_v = v_info
                        synth_target_c = None 
                        active_syllable_name = f"[ {v_info[0]} ]"
                        if operating_mode == 2:
                            synth.trigger_syllable_preset(None, v_info)
                        else:
                            synth.clear_gesture()

                    elif event.key in CONSONANTS:
                        c_info = CONSONANTS[event.key]
                        v_info = VOWELS[active_vowel_key]
                        synth_target_c = c_info
                        synth_target_v = v_info
                        s_name = f"{c_info['name']}{v_info[0]}"
                        active_syllable_name = f"[ {s_name} ]"
                        if operating_mode == 2:
                            synth.trigger_syllable_preset(c_info, v_info)
                        else:
                            synth.clear_gesture()

            frame = engine.get_frame()
            node = frame.fcz_macro
            ax = node.gamepad_axes
            active_iplv = node.iplv_32 
            live_theta_hz = frame.theta_freq

            # ==================================================================
            # 🚀 СКОРОСТНОЙ АВТО-ТЕСТ, ВЕДОМЫЙ ФАЗОЙ ТЕТЫ (THETA-SLAVED)
            # ==================================================================
            cur_th_norm = (synth.theta_phase / (2.0 * math.pi)) % 1.0
            if auto_test_running:
                if cur_th_norm < 0.10 and not auto_test_latched:
                    auto_test_latched = True
                    c_key = c_keys_list[auto_test_c_idx]
                    v_key = v_keys_list[auto_test_v_idx]
                    c_info = CONSONANTS[c_key]
                    v_info = VOWELS[v_key]
                    synth_target_c = c_info
                    synth_target_v = v_info
                    s_name = f"{c_info['name']}{v_info[0]}"
                    active_syllable_name = f"АВТО-ТЕСТ: [ {s_name} ]"
                    
                    if operating_mode == 2:
                        synth.trigger_syllable_preset(c_info, v_info)
                    else:
                        synth.clear_gesture()
                    
                    auto_test_v_idx += 1
                    if auto_test_v_idx >= len(v_keys_list):
                        auto_test_v_idx = 0
                        auto_test_c_idx += 1
                        if auto_test_c_idx >= len(c_keys_list):
                            auto_test_running = False
                            active_syllable_name = "АВТО-ТЕСТ ЗАВЕРШЕН!"
                            
                elif cur_th_norm > 0.50:
                    auto_test_latched = False

            if operating_mode == 0:
                with torch.inference_mode():
                    iplv_gpu = torch.from_numpy(active_iplv).to(DEVICE)
                    Vx = torch.sum(iplv_gpu * DX_GPU, dim=1, keepdim=True)
                    Vy = torch.sum(iplv_gpu * DY_GPU, dim=1, keepdim=True)
                    th_all_gpu = (torch.atan2(Vy, Vx) % (2.0 * math.pi)).squeeze(1).cpu().numpy()
                    ph_all_gpu = (torch.atan2(torch.full_like(Vx, ax.rx), torch.full_like(Vy, -ax.ry)) % (2.0 * math.pi)).squeeze(1).cpu().numpy()

                head_th, head_ph = th_all_gpu[-1], ph_all_gpu[-1]
                glide_speed = 3.8 * (1.0 + max(0.0, ax.ry))
                avatar_theta += ((head_th - avatar_theta + math.pi) % (2.0 * math.pi) - math.pi) * glide_speed * dt
                avatar_phi   += ((head_ph - avatar_phi + math.pi) % (2.0 * math.pi) - math.pi) * glide_speed * dt
                avatar_theta = avatar_theta % (2.0 * math.pi)
                avatar_phi   = avatar_phi % (2.0 * math.pi)
                synth.update_state(active_iplv, live_theta_hz, ax.rx, ax.ry, avatar_theta, avatar_phi)

            elif operating_mode == 1:
                pac_matrix, rx_sim, ry_sim, v_th, v_ph = generate_syllable_pac_tensor(synth_target_c, synth_target_v)
                active_iplv = pac_matrix
                live_theta_hz = 5.5
                avatar_theta = v_th
                avatar_phi = v_ph
                synth.update_state(active_iplv, live_theta_hz, rx_sim, ry_sim, avatar_theta, avatar_phi)

            elif operating_mode == 2:
                avatar_theta = synth.avatar_th
                avatar_phi = synth.avatar_ph
                synth.update_state(active_iplv, live_theta_hz, ax.rx, ax.ry, avatar_theta, avatar_phi)

            with torch.inference_mode():
                iplv_gpu = torch.from_numpy(active_iplv).to(DEVICE)
                Vx_all = torch.sum(iplv_gpu * DX_GPU, dim=1, keepdim=True)
                Vy_all = torch.sum(iplv_gpu * DY_GPU, dim=1, keepdim=True)
                th_all = (torch.atan2(Vy_all, Vx_all) % (2.0 * math.pi)).squeeze(1).cpu().numpy()
                ph_all = (torch.atan2(torch.full_like(Vx_all, ax.rx), torch.full_like(Vy_all, -ax.ry)) % (2.0 * math.pi)).squeeze(1).cpu().numpy()

            screen.fill((10, 12, 18))

            for m in range(6):
                ph = m * (2.0 * math.pi / 6.0)
                pts = [project_3d(*torus_to_3d(t, ph), yaw=torus_yaw) for t in np.linspace(0, 2.0*math.pi, 60)]
                if len(pts) > 1:
                    pygame.draw.lines(screen, (35, 25, 45), True, [p[:2] for p in pts], 1)
                for i in range(len(pts)-1):
                    if pts[i][2] > 0 and pts[i+1][2] > 0:
                        pygame.draw.line(screen, (90, 50, 120), pts[i][:2], pts[i+1][:2], 1)

            for th_val in np.linspace(0, 2.0*math.pi, 12, endpoint=False):
                pts = [project_3d(*torus_to_3d(th_val, p), yaw=torus_yaw) for p in np.linspace(0, 2.0*math.pi, 30)]
                pygame.draw.lines(screen, (25, 22, 35), True, [p[:2] for p in pts], 1)

            for l_th, l_ph, label, base_col in TORUS_LANDMARKS:
                tx, ty, tz = project_3d(*torus_to_3d(l_th, l_ph), yaw=torus_yaw)
                depth_scale = max(0.35, min(1.0, (tz + 180.0) / 360.0))
                col = (int(base_col[0] * depth_scale), int(base_col[1] * depth_scale), int(base_col[2] * depth_scale))
                radius = 4 if tz > 0 else 3
                pygame.draw.circle(screen, col, (tx, ty), radius)
                screen.blit(font_sm.render(label, True, col), (tx + 6, ty - 6))

            asx, asy, asz = project_3d(*torus_to_3d(avatar_theta, avatar_phi), yaw=torus_yaw)
            a_scale = max(0.4, min(1.0, (asz + 180.0) / 360.0))
            a_col = (int(255 * a_scale), int(50 * a_scale), int(150 * a_scale))
            pygame.draw.circle(screen, a_col, (asx, asy), int(12 * a_scale))
            pygame.draw.circle(screen, (255, 255, 255), (asx, asy), max(2, int(4 * a_scale)))

            traj_3d = []
            for k in range(32):
                px, py, pz = project_3d(*torus_to_3d(th_all[k], ph_all[k]), yaw=torus_yaw)
                traj_3d.append((px, py, pz, th_all[k]))

            for k in range(31):
                p1, p2 = traj_3d[k], traj_3d[k+1]
                if abs(p1[3] - p2[3]) < math.pi:
                    prog = k / 31.0
                    col = (int(255 * prog), int(155 * (1 - prog) + 100), 200)
                    thickness = max(1, int(1 + prog * 5))
                    pygame.draw.line(screen, col, p1[:2], p2[:2], thickness)
                    pygame.draw.circle(screen, col, p1[:2], max(1, int(1 + prog * 4)))

            cur_f1, cur_f2, target_voicing, live_th_dsp, cur_th_rad, cur_ph_rad = 500.0, 1500.0, 1.0, 5.5, 0.0, 0.0
            fft_log = None
            detected_eeg_syllable = None
            try:
                while not telemetry_queue.empty():
                    cur_f1, cur_f2, target_voicing, live_th_dsp, cur_th_rad, cur_ph_rad, fft_log, detected_eeg_syllable = telemetry_queue.get_nowait()
            except queue.Empty:
                pass

            if detected_eeg_syllable:
                add_to_chat(detected_eeg_syllable)

            if fft_log is not None:
                spec_surface.scroll(-2, 0)
                for y in range(128):
                    color = map_color(fft_log[y])
                    pygame.draw.line(spec_surface, color, (SPEC_W - 2, SPEC_H - 1 - y * (SPEC_H/128.0)), (SPEC_W, SPEC_H - 1 - y * (SPEC_H/128.0)), 2)

            PANEL_X = 850
            ty = 10

            # 1. 120-EDGE PHYSICAL MATRIX TELEMETRY
            screen.blit(font_lg.render("STREAMING 120-EDGE MATRIX", True, (0, 200, 255)), (PANEL_X, ty))
            pygame.draw.rect(screen, (20, 25, 35), (PANEL_X, ty + 18, 400, 80))
            pygame.draw.rect(screen, (0, 100, 150), (PANEL_X, ty + 18, 400, 80), 1)
            
            mean_iplv = np.mean(np.abs(active_iplv), axis=0)
            max_v = np.max(mean_iplv) + 1e-6
            for i in range(120):
                val = mean_iplv[i] / max_v
                h = int(val * 70)
                if i in IDX_CORE_SET: col = (255, 80, 80)     
                elif i in IDX_OUTER_SET: col = (80, 255, 120) 
                else: col = (80, 150, 255)                    
                pygame.draw.rect(screen, col, (PANEL_X + 10 + i*3, ty + 18 + 75 - h, 2, h))

            # 2. INTERACTIVE CV MATRIX CONTROL
            mid_y = ty + 105
            screen.blit(font_lg.render("INTERACTIVE CV MATRIX CONTROL", True, (255, 100, 200)), (PANEL_X, mid_y))
            pygame.draw.rect(screen, (35, 20, 25), (PANEL_X, mid_y + 18, 400, 115))
            pygame.draw.rect(screen, (150, 50, 100), (PANEL_X, mid_y + 18, 400, 115), 1)

            v_labels = [("[F1:И]", pygame.K_F1), ("[F2:Э]", pygame.K_F2), ("[F3:А]", pygame.K_F3), 
                        ("[F4:О]", pygame.K_F4), ("[F5:У]", pygame.K_F5), ("[F6:Ы]", pygame.K_F6),
                        ("[F7:Я]", pygame.K_F7), ("[F8:Ю]", pygame.K_F8)]
            for vi, (vl, vk) in enumerate(v_labels):
                col = (0, 255, 200) if vk == active_vowel_key else (120, 120, 140)
                bg_col = (0, 80, 60) if vk == active_vowel_key else (40, 30, 40)
                bx = PANEL_X + 12 + (vi % 4) * 96
                by = mid_y + 22 + (vi // 4) * 20
                pygame.draw.rect(screen, bg_col, (bx, by, 90, 17))
                pygame.draw.rect(screen, col, (bx, by, 90, 17), 1)
                screen.blit(font_sm.render(vl, True, col), (bx + 4, by + 2))

            c_labels = ["1:Т", "2:К", "3:Б", "4:С", "5:Ш", "6:М", "7:Д", "8:Г", "9:П", "0:Н", "-:НЬ"]
            for ci, cl in enumerate(c_labels):
                bx = PANEL_X + 12 + ci * 35
                by = mid_y + 65
                pygame.draw.rect(screen, (30, 30, 50), (bx, by, 32, 17))
                pygame.draw.rect(screen, (100, 150, 255), (bx, by, 32, 17), 1)
                screen.blit(font_sm.render(cl, True, (200, 220, 255)), (bx + 2, by + 2))

            screen.blit(font_lg.render(f"ACTIVE: {active_syllable_name}", True, (255, 255, 100)), (PANEL_X + 15, mid_y + 88))

            # 3. REAL-TIME SPECTROGRAM
            SPEC_Y = mid_y + 140
            screen.blit(font_lg.render("REAL-TIME SPECTROGRAM (0-11 kHz)", True, (200, 200, 200)), (PANEL_X, SPEC_Y))
            screen.blit(spec_surface, (PANEL_X, SPEC_Y + 18))
            pygame.draw.rect(screen, (100, 100, 100), (PANEL_X, SPEC_Y + 18, SPEC_W, SPEC_H), 1)

            # 4. DECODED SPEECH STREAM (LIVE CHAT)
            CHAT_Y = SPEC_Y + 110
            screen.blit(font_lg.render("DECODED SPEECH STREAM (LIVE CHAT)", True, (0, 255, 200)), (PANEL_X, CHAT_Y))
            pygame.draw.rect(screen, (15, 20, 28), (PANEL_X, CHAT_Y + 18, 400, 80))
            pygame.draw.rect(screen, (0, 200, 150), (PANEL_X, CHAT_Y + 18, 400, 80), 1)

            words = chat_syllables[-30:]
            line1 = " ".join(words[-30:-20]) if len(words) > 20 else ""
            line2 = " ".join(words[-20:-10]) if len(words) > 10 else ""
            line3 = " ".join(words[-10:]) if len(words) > 0 else "..."

            screen.blit(font_sm.render(f"> {line1}", True, (120, 140, 160)), (PANEL_X + 12, CHAT_Y + 22))
            screen.blit(font_sm.render(f"> {line2}", True, (180, 200, 220)), (PANEL_X + 12, CHAT_Y + 38))
            screen.blit(font_chat.render(f"> {line3}", True, (255, 240, 80)), (PANEL_X + 12, CHAT_Y + 58))

            # ==================================================================
            # 5. 🧬 BIOLOGICAL THETA & SPEECH RATE TELEMETRY (ТЕЛЕМЕТРИЯ СКОРОСТИ)
            # ==================================================================
            TELEM_Y = CHAT_Y + 105
            screen.blit(font_lg.render("BIOLOGICAL PAC TEMPO (WM 2.0)", True, (255, 180, 50)), (PANEL_X, TELEM_Y))
            pygame.draw.rect(screen, (22, 18, 28), (PANEL_X, TELEM_Y + 18, 400, 88))
            pygame.draw.rect(screen, (200, 120, 50), (PANEL_X, TELEM_Y + 18, 400, 88), 1)

            syllables_per_sec = live_th_dsp
            syllables_per_min = live_th_dsp * 60.0
            equivalent_bpm = live_th_dsp * 30.0
            syllable_dur_ms = 1000.0 / max(0.1, live_th_dsp)
            gamma_slot_dt = syllable_dur_ms / 32.0

            screen.blit(font_med.render(f"• THETA ПЕЙСМЕЙКЕР : {live_th_dsp:.2f} Hz  ({syllable_dur_ms:.0f} мс / слог)", True, (255, 255, 255)), (PANEL_X + 12, TELEM_Y + 24))
            screen.blit(font_med.render(f"• СКОРОСТЬ РЕЧИ    : {syllables_per_sec:.1f} слогов/сек | {syllables_per_min:.0f} слогов/мин", True, (0, 255, 200)), (PANEL_X + 12, TELEM_Y + 44))
            screen.blit(font_sm.render(f"• ЭКВИВАЛЕНТ ТЕМПА : {equivalent_bpm:.0f} BPM (16-дольный Psytrance)", True, (255, 100, 200)), (PANEL_X + 12, TELEM_Y + 64))
            screen.blit(font_xs.render(f"• 32 GAMMA BINS    : {gamma_slot_dt:.2f} мс на квант рабочей памяти", True, (150, 160, 180)), (PANEL_X + 12, TELEM_Y + 82))

            # Footer
            ui_y = HEIGHT - 45
            pygame.draw.rect(screen, (15, 12, 22), (0, ui_y, WIDTH, 45))
            pygame.draw.line(screen, (0, 255, 200), (0, ui_y), (WIDTH, ui_y), 2)
            screen.blit(font_med.render(f"NEUROCANVAS v35.0 | {MODE_NAMES[operating_mode]} | {live_th_dsp:.2f} Hz ({syllables_per_min:.0f} syl/min) | [SPACE] Авто-Тест", True, (255, 255, 255)), (20, ui_y + 14))

            pygame.display.flip()

    finally:
        stop_event.set()
        t_audio.join()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
