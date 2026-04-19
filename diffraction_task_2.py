import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from enum import Enum
from dataclasses import dataclass
import streamlit as st

# constants
NM=1e-9
MM=1e-3

# aperture types
class ApertureType(Enum):
    SINGLE_SLIT='Одиночная щель'
    DOUBLE_SLIT='Двойная щель'
    CIRCULAR='Круглое отверстие'
    RECTANGULAR='Прямоугольное отверстие'
    SQUARE_OBSTACLE='Непрозрочный экран'
    TRIANGLE='Треугольное отверстие'
    DIFFRACTION_GRATING='Дифракционная решетка'

# STANDART EXAMPLES - PRESETS

@dataclass
class Preset:
    name: str
    wavelength_nm: float
    a: float
    b: float
    aperture: ApertureType
    params: dict
    screen_size_mm: float
    grid_points: int

PRESETS = [
    Preset(
        name="Фраунгофер: щель 0.1 мм, λ=632 нм",
        wavelength_nm=632.8,
        a=1.0,
        b=2.0,
        aperture=ApertureType.SINGLE_SLIT,
        params={"width": 0.1e-3},
        screen_size_mm=20.0,
        grid_points=512,
    ),
    Preset(
        name="Френель: круглое отверстие 1 мм, λ=532 нм",
        wavelength_nm=532.0,
        a=0.3,
        b=0.3,
        aperture=ApertureType.CIRCULAR,
        params={"radius": 1.0e-3},
        screen_size_mm=10.0,
        grid_points=512,
    ),
    Preset(
        name="Фраунгофер: двойная щель, λ=650 нм",
        wavelength_nm=650.0,
        a=1.0,
        b=3.0,
        aperture=ApertureType.DOUBLE_SLIT,
        params={"slit_width": 0.05e-3, "slit_separation": 0.2e-3},
        screen_size_mm=30.0,
        grid_points=512,
    ),
    Preset(
        name="Френель: прямоугольное отверстие, λ=500 нм",
        wavelength_nm=500.0,
        a=0.5,
        b=0.5,
        aperture=ApertureType.RECTANGULAR,
        params={"width_x": 1.5e-3, "width_y": 0.5e-3},
        screen_size_mm=15.0,
        grid_points=512,
    ),
    Preset(
        name="Фраунгофер: круглое отверстие (диск Эйри), λ=550 нм",
        wavelength_nm=550.0,
        a=10.0,
        b=10.0,
        aperture=ApertureType.CIRCULAR,
        params={"radius": 0.5e-3},
        screen_size_mm=10.0,
        grid_points=512,
    ),
    Preset(
        name="Фраунгофер: дифракционная решетка, λ=550 нм",
        wavelength_nm=550.0,
        a=1.0,
        b=2.0,
        aperture=ApertureType.DIFFRACTION_GRATING,
        params={"period": 0.1e-3, "duty_cycle": 0.5},
        screen_size_mm=20.0,
        grid_points=512,
    ),
]

# CREATING APERTURE MASKS
def make_single_slit_mask(N,size,width):
    '''single slit mask (vertival)'''
    x=np.linspace(-size/2,size/2,N)
    mask=np.abs(x[np.newaxis,:])<=width/2
    return np.ones((N,N))*mask

def make_double_slit_mask(N,size,slit_width,slit_separation):
    '''double slit'''
    x=np.linspace(-size/2,size/2,N)
    center1=-slit_separation/2
    center2=slit_separation/2
    mask1=np.abs(x[np.newaxis,:]-center1)<=slit_width/2
    mask2=np.abs(x[np.newaxis,:]-center2)<=slit_width/2
    return np.ones((N,N))*(mask1|mask2)

def make_circular_mask(N,size,radius):
    '''circular hole mask'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    return (X**2+Y**2)<=radius**2

def make_rectangular_mask(N,size,width_x,width_y):
    '''rectangular mask'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    return (np.abs(X)<=width_x/2) &(np.abs(Y)<=width_y/2)

def make_square_obstacle_mask(N,size,radius):
    '''NOT CLEAR disk on a clear background'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    return ~((X**2+Y**2)<=radius**2) # wavy thing is used to negate so that true -> false

def make_triangle_mask(N,size):
    '''sides equal rectangle'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    h=size*0.4
    v0=np.array([0,h/2])
    v1=np.array([-h/np.sqrt(3),-h/2])
    v2=np.array([h/np.sqrt(3),-h/2])

    def sign(p1,p2,p3):
        mat = np.array([[p1[0], p1[1], 1],
                 [p2[0], p2[1], 1],
                 [p3[0], p3[1], 1]])
        return np.linalg.det(mat)
    points=np.column_stack([X.ravel(),Y.ravel()])
    d1=sign(points,v0,v1)
    d2=sign(points,v1,v2)
    d3=sign(points,v2,v0)
    has_neg=(d1<0)|(d2<0)|(d3<0)
    has_pos=(d1>0)|(d2>0)|(d3>0)
    mask=~(has_neg&has_pos)
    return mask.reshape(N,N)

def make_diffraction_grating_mask(N,size,period,duty_cycle):
    '''diffraction grating mask'''
    x=np.linspace(-size/2,size/2,N)
    y=np.linspace(-size/2,size/2,N)
    X,Y=np.meshgrid(x,y)
    slit_width=period*duty_cycle
    mask=(X%period)<=slit_width
    return mask

def make_aperture(aperture_type,N,aperture_size,params):
    '''Creating aperture mask of a given type'''
    if aperture_type == ApertureType.SINGLE_SLIT:
        w = params.get("width", aperture_size / 10)
        return make_single_slit_mask(N, aperture_size, w)
    elif aperture_type == ApertureType.DOUBLE_SLIT:
        sw = params.get("slit_width", aperture_size / 20)
        ss = params.get("slit_separation", aperture_size / 5)
        return make_double_slit_mask(N, aperture_size, sw, ss)
    elif aperture_type == ApertureType.CIRCULAR:
        r = params.get("radius", aperture_size / 6)
        return make_circular_mask(N, aperture_size, r)
    elif aperture_type == ApertureType.RECTANGULAR:
        wx = params.get("width_x", aperture_size / 4)
        wy = params.get("width_y", aperture_size / 8)
        return make_rectangular_mask(N, aperture_size, wx, wy)
    elif aperture_type == ApertureType.SQUARE_OBSTACLE:
        r = params.get("radius", aperture_size / 10)
        return make_square_obstacle_mask(N, aperture_size, r)
    elif aperture_type == ApertureType.TRIANGLE:
        return make_triangle_mask(N, aperture_size)
    elif aperture_type == ApertureType.DIFFRACTION_GRATING:
        period = params.get("period", aperture_size / 20)
        duty = params.get("duty_cycle", 0.5)
        return make_diffraction_grating_mask(N, aperture_size, period, duty)
    else:
        return np.ones((N, N))
    
# CALCULATING FRENsEL DIFFRACTION
def frensel_diffraction(aperture,wavelength,a,b,aperture_size,screen_size,N):
    '''
    numerical evaluation of frensel diffraction using FFT'''
    k=2*np.pi/wavelength

    x=np.linspace(-screen_size/2,screen_size/2,N)
    y=np.linspace(-screen_size/2,screen_size/2,N)
    dx=x[1]-x[0]
    dy=y[1]-y[0]
    X,Y=np.meshgrid(x,y)

    xi=x
    eta=y
    XI,ETA=np.meshgrid(xi,eta)

    phase_screen =np.exp(1j*k*(X**2+Y**2)/(2*b))
    phase_b=np.exp(1j*k*(XI**2+ETA**2)/(2*b))

    if np.isfinite(a) and a>0:
        phase_a=np.exp(1j*k*(XI**2+ETA**2)/(2*b))
    else:
        phase_a=np.ones_like(XI)

    if aperture.shape!=(N,N):
        from scipy.ndimage import zoom
        zy,zx=N/aperture.shape[0],N/aperture.shape[1]
        aperture_resampled=zoom(aperture.real,(zx,zy),order=1)
        aperture_resampled=(aperture_resampled>0.5).astype(complex)
    else:
        aperture_resampled=aperture.astype(complex)
    U_eff=aperture_resampled*phase_b*phase_a

    prefactor=np.exp(1j*k*b)/(1j*wavelength*b)
    U_fft=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_eff)))
    U_fft*=dx*dy
    
    U=prefactor*phase_screen*U_fft

    intensity=np.abs(U)**2

    max_I=np.max(intensity)
    if max_I>0:
        intensity/=max_I
    return intensity

# FRAUNHOFER DIFFRACTION 
def fraunhofer_diffraction(aperture,wavelength,b,aperture_size,screen_size,N):
    '''Fraunhofer calculation (far away) using fft once again boring'''
    xi=np.linspace(-aperture_size/2,aperture_size/2,N)
    eta=np.linspace(-aperture_size/2,aperture_size/2,N)
    dxi=xi[1]-xi[0]
    deta=eta[1]-eta[0]

    A_fft=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))

    x_nat=np.fft.fftshift(np.fft.fftfreq(N,dxi))*wavelength*b
    y_nat=np.fft.fftshift(np.fft.fftfreq(N,deta))*wavelength*b

    intensity=np.abs(A_fft)**2
    max_I=np.max(intensity)
    if max_I>0:
        intensity/=max_I
    return intensity,x_nat,y_nat

# AUTO SELECTION BETWEEN CALC TYPES FOR CHRIST SAKE 
def compute_diffraction(aperture,wavelength,a,b,aperture_size,screen_size,N,mode='auto'):
    '''
    mode="frensel"/"fraunhofer"/"auto" 
    '''
    d_char=aperture_size/3
    N_F=d_char**2/(wavelength*b)
    if mode=='auto':
            mode='frensel' if N_F>0.5 else 'fraunhofer'
    if mode=='frensel':
        intensity=frensel_diffraction(
            aperture,wavelength,a,b,aperture_size,screen_size,N)
        x=np.linspace(-screen_size/2,screen_size/2,N)
        y=np.linspace(-screen_size/2,screen_size/2,N)
        return intensity,x,y,'frensel',N_F
    else:
        intensity,x,y=fraunhofer_diffraction(
            aperture,wavelength,b,aperture_size,screen_size,N
        )
        return intensity,x,y,'fraunhofer',N_F
    
# STREAMLIT VISUAL FOR PICKMES
def main():
    st.set_page_config(page_title='Дифракция Френеля и Фраунгофера',layout='wide')
    st.title('Дифракция Френеля и Фраунгофера')

    col1,col2=st.columns([1,2])
    
    with col1:
        st.subheader('Параметры')

        preset_names=['---Выбор Пресета---']+[p.name for p in PRESETS]
        selecred_preset=st.selectbox('Пресет',preset_names,index=0)
        if selecred_preset!='---Выбор Пресета--':
            preset_idx=preset_names.index(selecred_preset)-1
            p=PRESETS[preset_idx]
            default_wavelength=p.wavelength_nm
            default_a=p.a
            default_b=p.b
            default_aperture_size=p.screen_size_mm
            default_screen_size=p.screen_size_mm
            default_grid=p.grid_points
            default_aperture_type=p.aperture
            default_params={k:v*1000 for k,v in p.params.items()}
        else:
            default_wavelength=532.0
            default_a=1.0
            default_b=1.0
            default_aperture_size=5.0
            default_screen_size=20.0
            default_aperture_type=ApertureType.SINGLE_SLIT
            default_params={}

        wavelength_nm = st.number_input("Длина волны (нм)", value=default_wavelength, min_value=1.0)
        a = st.number_input("Расст. источник → отверстие (м)", value=default_a, min_value=0.01)
        b = st.number_input("Расст. отверстие → экран (м)", value=default_b, min_value=0.01)
        aperture_size_mm = st.number_input("Размер апертуры (мм)", value=default_aperture_size, min_value=0.1)
        screen_size_mm = st.number_input("Размер экрана (мм)", value=default_screen_size, min_value=0.1)
        grid_points = st.number_input("Точки сетки", value=default_grid, min_value=64, max_value=2048, step=64)

        st.subheader("Тип апертуры")
        aperture_type=st.selectbox(
            'aperture type',
            [t.value for t in ApertureType],
            index=list(ApertureType).index(default_aperture_type)
        )
        ap_type=ApertureType(aperture_type)

        st.subheader('Параметры Апертруры')
        params={}

        if ap_type == ApertureType.SINGLE_SLIT:
            params["width"] = st.number_input("Ширина щели (мм)", value=default_params.get("width", 0.1), min_value=0.001)
        elif ap_type == ApertureType.DOUBLE_SLIT:
            params["slit_width"] = st.number_input("Ширина щели (мм)", value=default_params.get("slit_width", 0.05), min_value=0.001)
            params["slit_separation"] = st.number_input("Расстояние между щелями (мм)", value=default_params.get("slit_separation", 0.2), min_value=0.001)
        elif ap_type == ApertureType.CIRCULAR:
            params["radius"] = st.number_input("Радиус (мм)", value=default_params.get("radius", 0.5), min_value=0.001)
        elif ap_type == ApertureType.RECTANGULAR:
            params["width_x"] = st.number_input("Ширина X (мм)", value=default_params.get("width_x", 1.5), min_value=0.001)
            params["width_y"] = st.number_input("Ширина Y (мм)", value=default_params.get("width_y", 0.5), min_value=0.001)
        elif ap_type == ApertureType.SQUARE_OBSTACLE:
            params["radius"] = st.number_input("Радиус диска (мм)", value=default_params.get("radius", 0.5), min_value=0.001)
        elif ap_type == ApertureType.DIFFRACTION_GRATING:
            params["period"] = st.number_input("Период решётки (мм)", value=default_params.get("period", 0.05), min_value=0.001)
            params["duty_cycle"] = st.slider("Заполнение (0-1)", 0.1, 0.9, default_params.get("duty_cycle", 0.5))
        
        st.subheader('Режим Рассчета')
        mode=st.radio('Режим',['auto','frensel','fraunhofer'],horizontal=True)

        params_key=(
            wavelength_nm,a,b,aperture_size_mm,screen_size_mm,grid_points,ap_type,tuple(sorted(params.items())))
        
    
    with col2:
        if True:
            try:
                wavelength=wavelength_nm*NM
                aperture_size=aperture_size_mm*MM
                screen_size=screen_size_mm*MM
                N=int(grid_points)

                params_m={k: v*MM for k,v in params.items()}
                aperture=make_aperture(ap_type,N,aperture_size,params_m)

                intensity,x,y,mode_used,N_F=compute_diffraction(aperture,wavelength,a,b,aperture_size,screen_size,N,mode=mode)

                fig=plt.figure(figsize=(12.8, 8.0))
                gs=GridSpec(2,2,fig,hspace=0.3,wspace=0.3)

                ax1 = fig.add_subplot(gs[0, 0])
                extent = [-aperture_size_mm / 2, aperture_size_mm / 2,
                          -aperture_size_mm / 2, aperture_size_mm / 2]
                ax1.imshow(aperture, cmap="gray", extent=extent, origin="lower")
                ax1.set_xlabel("мм")
                ax1.set_ylabel("мм")
                ax1.set_title("Апертура")
                ax1.set_aspect("equal")

                ax2 = fig.add_subplot(gs[0, 1])
                extent = [-screen_size_mm / 2, screen_size_mm / 2,
                          -screen_size_mm / 2, screen_size_mm / 2]
                im = ax2.imshow(intensity, cmap="hot", extent=extent, origin="lower", vmin=0, vmax=1)
                ax2.set_xlabel("мм")
                ax2.set_ylabel("мм")
                mode_label = "Френель" if mode_used == "frensel" else "Фраунгофер"
                ax2.set_title(f"Интенсивность ({mode_label}, N_F={N_F:.2f})")
                fig.colorbar(im, ax=ax2, label="I / I_max")

                ax3 = fig.add_subplot(gs[1, :])
                mid = len(y) // 2
                x_mm = np.linspace(-screen_size_mm / 2, screen_size_mm / 2, len(x))
                ax3.plot(x_mm, intensity[mid, :], "b-", linewidth=1)
                ax3.set_xlabel("Положение на экране (мм)")
                ax3.set_ylabel("I / I_max")
                ax3.set_title("Профиль интенсивности (центральное сечение)")
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(-screen_size_mm / 2, screen_size_mm / 2)

                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка при расчёте: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    print(' run python -m streamlit run diffraction_task_2.py')
    main()