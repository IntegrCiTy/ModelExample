import numpy as np
import pandas as pd
from scipy.integrate import odeint


def dtok(temp):
    """Transform C.deg into K.deg"""
    return temp + 273.15


def ktod(temp):
    """Transform K.deg into C.deg"""
    return temp - 273.15


def vol_cst_model(y, time, m_dot_src, t_src, m_dot_snk, t_snk, ex_surf, mesh_h):
    """Define ODE for a 3 mesh constant volume stratified thermal buffer"""
    t_top, t_mid, t_bot = y

    mesh_vol = ex_surf * mesh_h

    rho = 985  # kg.m-3
    cp = 4185  # J.kg-1.K-1
    e_cond = 0.62  # W.m-1.K-1

    phi_port_top = m_dot_src * cp * t_src - m_dot_snk * cp * t_top
    phi_port_low = m_dot_snk * cp * t_snk - m_dot_src * cp * t_bot

    phi_cond_top_to_mid = e_cond * mesh_h * ex_surf * (t_top - t_mid)
    phi_cond_mid_to_bot = e_cond * mesh_h * ex_surf * (t_mid - t_bot)

    dydt = [
        phi_port_top - phi_cond_top_to_mid,
        phi_cond_top_to_mid - phi_cond_mid_to_bot,
        phi_port_low + phi_cond_mid_to_bot
    ]

    m_dot_mix = m_dot_snk - m_dot_src

    if m_dot_mix > 0:  # m_dot_mix UP
        dydt[0] += m_dot_mix * cp * t_mid
        dydt[1] += m_dot_mix * cp * (t_bot - t_mid)
        dydt[2] -= m_dot_mix * cp * t_bot

    if m_dot_mix < 0:  # m_dot_mix DOWN
        dydt[0] += m_dot_mix * cp * t_top
        dydt[1] -= m_dot_mix * cp * (t_top - t_mid)
        dydt[2] -= m_dot_mix * cp * t_mid

    return np.array(dydt) / (mesh_vol * rho * cp)


def tau_model(y, t, io, tau, p_set):
    """Define ODE for a simplified dynamic heat pump model using time constant"""
    dydt = (p_set * io - y) / tau
    return dydt


class ThermalBuffer:
    """Model class of a 3 mesh constant volume stratified thermal buffer"""

    UNIT = {"seconds": 1, "minutes": 60, "hours": 3600}

    def __init__(self, ex_surf, mesh_h, temp_init=None, start='1/1/2000'):
        if not temp_init:
            temp_init = [dtok(80.0), dtok(60.0), dtok(40.0)]
        assert len(temp_init) == 3

        self.m_dot_src = 0  # kg/s
        self.t_src = temp_init[0]  # K.deg
        self.m_dot_snk = 0  # kg/s
        self.t_snk = temp_init[2]  # K.deg

        self.ex_surf = ex_surf
        self.mesh_h = mesh_h

        self.time = pd.to_datetime(start)

        self.t_top = temp_init[0]
        self.t_mid = temp_init[1]
        self.t_bot = temp_init[2]

        self.t_set_top = temp_init[0]
        self.t_set_bop = temp_init[2]
        self.soc = (np.mean([self.t_top, self.t_mid, self.t_bot]) - self.t_set_bop) / (self.t_set_top - self.t_set_bop)

    def make_step(self, step, unit="seconds"):
        end = self.time + pd.DateOffset(**{unit: step})
        t = np.arange(start=0, stop=step * self.UNIT[unit], step=1.0)

        y0 = [self.t_top, self.t_mid, self.t_bot]
        res = odeint(vol_cst_model, y0, t, args=(
            self.m_dot_src,
            self.t_src,
            self.m_dot_snk,
            self.t_snk,
            self.ex_surf,
            self.mesh_h))

        self.t_top, self.t_mid, self.t_bot = res[-1]
        self.soc = (np.mean([self.t_top, self.t_mid, self.t_bot]) - self.t_set_bop) / (self.t_set_top - self.t_set_bop)

        self.time += pd.DateOffset(**{unit: step})


class HeatPump:
    """Model class of a HP dynamic model based on a ratio of the theoretical COP of Carnot"""

    UNIT = {"seconds": 1, "minutes": 60, "hours": 3600}

    def __init__(self, p_max, t_cond, t_evap, n_th=0.4, tau=60.0, io_init=False, start='1/1/2000'):
        assert t_cond > t_evap

        self.p_max = p_max
        self.t_cond = t_cond
        self.t_evap = t_evap

        self.n_th = n_th
        self.tau = tau

        self.time = pd.to_datetime(start)

        self.io = io_init

        self.cop = self.n_th * (self.t_cond / (self.t_cond - self.t_evap))

        self.p_sink = self.io * self.p_max
        self.p_elec = self.p_sink / self.cop
        self.p_srce = self.p_sink - self.p_elec

    def make_step(self, step, unit="seconds"):
        end = self.time + pd.DateOffset(**{unit: step})
        t = np.arange(start=0, stop=step * self.UNIT[unit], step=1.0)

        res_p_sink = odeint(tau_model, self.p_sink, t, args=(self.io, self.tau, self.p_max))

        self.cop = self.n_th * (self.t_cond / (self.t_cond - self.t_evap))

        self.p_sink = round(res_p_sink[-1][0], 3)
        self.p_elec = self.p_sink / self.cop
        self.p_srce = self.p_sink - self.p_elec

        self.time += pd.DateOffset(**{unit: step})


class Profile:
    def __init__(self, data, method="nearest"):
        self.series = data
        self.time = pd.to_datetime(data.index[0])
        self.value = data.iloc[0]

        self.method = method

    def make_step(self, step, unit="seconds"):
        self.time += pd.DateOffset(**{unit: step})
        self.value = self.series.iloc[self.series.index.get_loc(self.time, method=self.method)]


class Hysteresis:
    def __init__(self, x_max=1.0, x_min=0.0, y_init=False):
        self.x_max = x_max
        self.x_min = x_min

        self.x = None
        self.y = y_init

    def make_step(self):
        if self.y and self.x >= self.x_max:
            self.y = False
        if not self.y and self.x <= self.x_min:
            self.y = True


def t_out_from_m_dot_and_p(t_in, m_dot, p_kw):
    cp = 4.18  # kJ/kg/K
    return p_kw / m_dot / cp + t_in


def m_dot_from_p_and_t_in_and_t_set(p_kw, t_in, t_set):
    cp = 4.18  # kJ/kg/K
    return p_kw / cp / abs(t_in - t_set)  # always >=0
