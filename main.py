import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import os
import time


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class GrahamScanConvexHull:
    def __init__(self):
        self.points = []
        self.lines = []
        self.convex_hull = []
        self.start_time = None
        self.end_time = None

    def read_obj_file(self, filename):
        """Lê pontos 2D de um arquivo .obj"""
        self.points = []
        self.lines = []
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):  # Vertex line
                        parts = line.split()
                        if len(parts) >= 3:
                            x, y = float(parts[1]), float(parts[2])
                            self.points.append(Point(x, y))
                    elif line.startswith('l '):  # Line definition
                        parts = line.split()
                        indices = [int(idx) for idx in parts[1:]]
                        self.lines.append(indices)
            return True
        except Exception as e:
            print(f"Erro ao ler arquivo: {e}")
            return False

    def polar_angle(self, p0, p1): # substituído pelo pseudo-ângulo abaixo
        """Calcula o ângulo polar do ponto p1 em relação a p0"""
        if p1.x == p0.x:
            return math.pi / 2 if p1.y > p0.y else -math.pi / 2
        return math.atan2(p1.y - p0.y, p1.x - p0.x)

    def pseudo_angle(self, p0, p1):
        """Calcula o pseudo-ângulo do ponto p1 em relação a p0."""
        dx = p1.x - p0.x
        dy = p1.y - p0.y

        if dx == 0 and dy == 0:
            return -1  # ponto coincidente

        if dx >= 0 and dy >= 0:
            return dy / (dx + dy)
        if dx <= 0 and dy >= 0:
            return 1 + abs(dx) / (abs(dx) + dy)
        if dx <= 0 and dy <= 0:
            return 2 + dy / (dx + dy)
        return 3 + dx / (dx + abs(dy))

    def distance_squared(self, p1, p2):
        """Calcula a distância ao quadrado entre dois pontos"""
        return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

    def cross_product(self, O, A, B):
        """Calcula o produto cruzado dos vetores OA e OB"""
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)

    # Como são coordenadas polares, ao invés de passar 2 vetores, dá pra passar só 3 pontos.
    # Fica mais simples considerando onde a função é chamada.
    # Fica mais rápido também (testei)

    def run_time(self):
        elapsed_time = self.end_time - self.start_time
        time_txt = "s"
        if (elapsed_time < 1):
            new_time = elapsed_time * 1000
            time_txt = "ms"
        if (new_time < 1):
            new_time = new_time * 1000
            time_txt = "µs"
        return f"{new_time:.4f} {time_txt}"

    def graham_scan(self):
        """Implementa o algoritmo Graham Scan"""
        self.start_time = time.perf_counter()
        if len(self.points) < 3:
            self.convex_hull = self.points[:]
            return self.convex_hull

        # Encontra o ponto com menor coordenada y (e menor x em caso de empate)
        start_point = min(self.points, key=lambda p: (p.y, p.x))

        # Ordena os pontos por ângulo polar em relação ao ponto inicial
        other_points = [p for p in self.points if p != start_point]

        def sort_key(point):
            #angle = self.polar_angle(start_point, point)
            angle = self.pseudo_angle(start_point, point)
            dist = self.distance_squared(start_point, point)
            return (angle, dist)

        other_points.sort(key=sort_key) # built-in do Python, Timsort Algorithm

        # Remove pontos colineares, mantendo apenas o mais distante
        unique_points = [start_point]
        for point in other_points:
            while (len(unique_points) > 1 and
                   abs(self.pseudo_angle(start_point, unique_points[-1]) -
                       self.pseudo_angle(start_point, point)) < 1e-9):
                   #abs(self.polar_angle(start_point, unique_points[-1]) -
                       #self.polar_angle(start_point, point)) < 1e-9):
                if (self.distance_squared(start_point, point) >
                        self.distance_squared(start_point, unique_points[-1])):
                    unique_points.pop()
                    #break
                    unique_points.append(point)
                else:
                    break
            else:
                unique_points.append(point)

        if len(unique_points) < 3:
            self.convex_hull = unique_points
            return self.convex_hull

        # Constrói o fecho convexo
        hull = []
        for point in unique_points:
            # Remove pontos que fazem curva à direita
            while (len(hull) > 1 and
                   self.cross_product(hull[-2], hull[-1], point) <= 0):
                hull.pop()
            hull.append(point)

        self.convex_hull = hull
        self.end_time = time.perf_counter()

        #print(f"Execution time: {self.run_time()[0]:.4f} {self.run_time()[1]}")
        return self.convex_hull

    def graham_scan_step_by_step(self):
        """Gerador que executa Graham Scan passo a passo."""
        if len(self.points) < 3:
            yield self.points[:], None, []
            return

        start_point = min(self.points, key=lambda p: (p.y, p.x))
        other_points = [p for p in self.points if p != start_point]

        def sort_key(p):
            angle = self.polar_angle(start_point, p)
            dist = self.distance_squared(start_point, p)
            return (angle, dist)

        other_points.sort(key=sort_key)

        unique_points = [start_point]
        for point in other_points:
            while (len(unique_points) > 1 and
                   abs(self.polar_angle(start_point, unique_points[-1]) - self.polar_angle(start_point, point)) < 1e-9):
                if self.distance_squared(start_point, point) > self.distance_squared(start_point, unique_points[-1]):
                    unique_points.pop()
                    # break
                    unique_points.append(point)
                else:
                    break
            else:
                unique_points.append(point)

        hull = []
        discarded = []
        for point in unique_points:
            while len(hull) > 1 and self.cross_product(hull[-2], hull[-1], point) <= 0:
                removed = hull.pop()
                discarded.append(removed)
                yield hull[:], point, discarded[:]
            hull.append(point)
            yield hull[:], point, discarded[:]

        self.convex_hull = hull
        yield hull[:], None, discarded[:]  # Fecho final

    def save_result(self, filename):
        """Salva o resultado em um arquivo .obj"""
        try:
            with open(filename, 'w') as file:
                file.write("# Fecho Convexo - Resultado do Graham Scan\n")
                file.write(f"# Total de pontos no Fecho: {len(self.convex_hull)}\n\n")

                # Escreve os vértices do fecho convexo
                for point in self.convex_hull:
                    file.write(f"v {point.x} {point.y} 0.0\n")

                # Escreve as arestas do fecho convexo
                file.write("\n# Arestas do fecho convexo\n")
                for i in range(len(self.convex_hull)):
                    next_i = (i + 1) % len(self.convex_hull)
                    file.write(f"l {i + 1} {next_i + 1}\n")

            return True
        except Exception as e:
            print(f"Erro ao salvar arquivo: {e}")
            return False


class ConvexHullGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Graham Scan - Fecho Convexo 2D")
        self.root.geometry("1000x700")

        self.graham_scan = GrahamScanConvexHull()
        self.setup_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.root.destroy()
        sys.exit()

    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Frame de controles
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Botões
        ttk.Button(control_frame, text="Carregar Arquivo .obj",
                   command=self.load_file).grid(row=0, column=0, padx=(0, 5))

        ttk.Button(control_frame, text="Calcular Fecho Convexo",
                   command=self.calculate_hull).grid(row=0, column=1, padx=(0, 5))

        ttk.Button(control_frame, text="Salvar Resultado",
                   command=self.save_result).grid(row=0, column=2, padx=(0, 5))

        ttk.Button(control_frame, text="Gerar Exemplo",
                   command=self.generate_example).grid(row=0, column=3, padx=(0, 5))

        ttk.Button(control_frame, text="Passo a Passo",
                   command=self.run_step_by_step).grid(row=0, column=4, padx=(0, 5))

        # Labels de informação
        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.points_label = ttk.Label(self.info_frame, text="Pontos Carregados: 0")
        self.points_label.grid(row=0, column=0, padx=(0, 20))

        self.hull_label = ttk.Label(self.info_frame, text="Pontos no Fecho: 0")
        self.hull_label.grid(row=0, column=1, padx=(0, 20))

        self.runtime_label = ttk.Label(self.info_frame, text="Execução: 0 s")
        self.runtime_label.grid(row=0, column=2)

        # Gráfico
        self.setup_plot()

        # Configurar redimensionamento
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def setup_plot(self):
        """Configura o gráfico matplotlib"""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2,
                                         sticky=(tk.W, tk.E, tk.N, tk.S),
                                         padx=10, pady=10)

        self.ax.set_title("Fecho Convexo - Graham Scan")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # Variáveis de estado para pan
        self._pan_start = None

        # Conectar eventos
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_scroll(self, event):
        """Zoom com a scroll wheel"""
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return  # Fora dos limites do gráfico

        new_xlim = [xdata + (x - xdata) * scale_factor for x in cur_xlim]
        new_ylim = [ydata + (y - ydata) * scale_factor for y in cur_ylim]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def on_press(self, event):
        """Inicia o pan ao clicar"""
        if event.button == 1:  # Botão esquerdo
            self._pan_start = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_release(self, event):
        """Finaliza o pan ao soltar o botão"""
        self._pan_start = None

    def on_motion(self, event):
        """Move a visualização durante o pan"""
        if self._pan_start is None or event.button != 1:
            return

        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        xlim = self._pan_start[2]
        ylim = self._pan_start[3]

        # Ajuste proporcional ao tamanho da figura
        widget = self.canvas.get_tk_widget()
        width = widget.winfo_width()
        height = widget.winfo_height()

        # Evitar divisão por zero
        if width == 0 or height == 0:
            return

        scale_x = (xlim[1] - xlim[0]) / width
        scale_y = (ylim[1] - ylim[0]) / height

        new_xlim = (xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
        new_ylim = (ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def load_file(self):
        """Carrega arquivo .obj"""
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo .obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )

        if filename:
            if self.graham_scan.read_obj_file(filename):
                self.points_label.config(text=f"Pontos Carregados: {len(self.graham_scan.points)}")
                self.hull_label.config(text="Pontos no Fecho: 0")
                self.plot_points()
                messagebox.showinfo("Sucesso", f"Arquivo carregado com {len(self.graham_scan.points)} pontos!")
            else:
                messagebox.showerror("Erro", "Erro ao carregar o arquivo!")

    def calculate_hull(self):
        """Calcula o fecho convexo"""
        if not self.graham_scan.points:
            messagebox.showwarning("Aviso", "Carregue um arquivo primeiro!")
            return

        hull = self.graham_scan.graham_scan()
        self.hull_label.config(text=f"Pontos no Fecho: {len(hull)}")
        self.runtime_label.config(text=f"Execução: {self.graham_scan.run_time()}")
        self.plot_points()

        messagebox.showinfo("Resultado",
                            f"Fecho convexo calculado!\n"
                            f"Pontos Originais: {len(self.graham_scan.points)}\n"
                            f"Pontos no Fecho: {len(hull)}\n"
                            f"Tempo de Execução: {self.graham_scan.run_time()}")

    def save_result(self):
        """Salva o resultado"""
        if not self.graham_scan.convex_hull:
            messagebox.showwarning("Aviso", "Calcule o fecho convexo primeiro!")
            return

        filename = filedialog.asksaveasfilename(
            title="Salvar resultado",
            defaultextension=".obj",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )

        if filename:
            if self.graham_scan.save_result(filename):
                messagebox.showinfo("Sucesso", "Resultado salvo com sucesso!")
            else:
                messagebox.showerror("Erro", "Erro ao salvar o arquivo!")

    def generate_example(self):
        """Gera um exemplo de pontos para teste"""
        # Gera pontos aleatórios
        np.random.seed()  # Para resultados reproduzíveis
        n_points = np.random.randint(3, 50) #20

        # Gera pontos em um círculo com alguns pontos internos
        angles = np.random.uniform(0, 2 * np.pi, n_points // 2)
        radii = np.random.uniform(0.5, 1.0, n_points // 2)
        x_circle = radii * np.cos(angles)
        y_circle = radii * np.sin(angles)

        # Adiciona alguns pontos internos
        x_internal = np.random.uniform(-0.5, 0.5, n_points // 2)
        y_internal = np.random.uniform(-0.5, 0.5, n_points // 2)

        x_all = np.concatenate([x_circle, x_internal])
        y_all = np.concatenate([y_circle, y_internal])

        # Cria os pontos
        self.graham_scan.points = []
        for x, y in zip(x_all, y_all):
            self.graham_scan.points.append(Point(x, y))

        self.points_label.config(text=f"Pontos Carregados: {len(self.graham_scan.points)}")
        self.hull_label.config(text="Pontos no Fecho: 0")
        self.graham_scan.convex_hull = []
        self.plot_points()

        messagebox.showinfo("Exemplo", f"Gerado exemplo com {len(self.graham_scan.points)} pontos!")

    def plot_points(self):
        """Plota os pontos e o fecho convexo"""
        self.ax.clear()

        if self.graham_scan.points:
            # Plota todos os pontos
            x_coords = [p.x for p in self.graham_scan.points]
            y_coords = [p.y for p in self.graham_scan.points]
            #self.ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Arestas originais')
            self.ax.scatter(x_coords, y_coords, c='blue', alpha=0.6, s=30, label='Pontos originais')

        if self.graham_scan.lines:
            for idx, line in enumerate(self.graham_scan.lines):
                if len(line) >= 2:
                    xs = [self.graham_scan.points[i - 1].x for i in line]
                    ys = [self.graham_scan.points[i - 1].y for i in line]
                    if idx == 0:
                        self.ax.plot(xs, ys, 'b-', linewidth=1, alpha=0.7, label='Arestas originais')
                    else:
                        self.ax.plot(xs, ys, 'b-', linewidth=1, alpha=0.7)

        if self.graham_scan.convex_hull:
            # Plota o fecho convexo
            hull_x = [p.x for p in self.graham_scan.convex_hull]
            hull_y = [p.y for p in self.graham_scan.convex_hull]

            # Adiciona o primeiro ponto no final para fechar o polígono
            hull_x.append(hull_x[0])
            hull_y.append(hull_y[0])

            self.ax.plot(hull_x, hull_y, 'r-', linewidth=2, label='Fecho Convexo')
            self.ax.scatter(hull_x[:-1], hull_y[:-1], c='red', s=50, zorder=5, label='Vértices do fecho')

        self.ax.set_title("Fecho Convexo - Graham Scan")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)
        #self.ax.legend()
        self.ax.set_aspect('equal')

        self.canvas.draw()

    def plot_step(self, hull, current_point, discarded, step):
        self.ax.clear()

        # Todos os pontos
        if self.graham_scan.points:
            x_coords = [p.x for p in self.graham_scan.points]
            y_coords = [p.y for p in self.graham_scan.points]
            self.ax.scatter(x_coords, y_coords, c='blue', alpha=0.6, s=30, label='Pontos')

        # Descartados
        if discarded:
            xd = [p.x for p in discarded]
            yd = [p.y for p in discarded]
            self.ax.scatter(xd, yd, c='gray', alpha=0.9, s=30, label='Descartados')

        # Hull parcial
        if len(hull) >= 2:
            xs = [p.x for p in hull]
            ys = [p.y for p in hull]

            # Fechar se for último passo
            if current_point is None and len(hull) > 2:
                xs.append(hull[0].x)
                ys.append(hull[0].y)

            self.ax.plot(xs, ys, 'r-', linewidth=2)

        # Ponto atual
        if current_point:
            self.ax.plot(current_point.x, current_point.y, 'go', markersize=10)

        self.ax.set_title(f"Fecho Convexo - Passo {step}")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        self.canvas.draw()
        self.canvas.flush_events()

    def run_step_by_step(self):
        """Executa o passo a passo com tempo configurável e destaca descartados."""
        if not self.graham_scan.points:
            messagebox.showwarning("Aviso", "Carregue um arquivo primeiro!")
            return

        delay = simpledialog.askfloat(
            "Configurar Delay",
            "Informe o tempo de espera entre os passos (em milisegundos):",
            minvalue=0.0, maxvalue=10000.0
        )

        if delay is None:
            return  # Cancelado

        delay_ms = int(delay)

        self.step_generator = self.graham_scan.graham_scan_step_by_step()
        self.step_count = 0

        def step():
            try:
                hull, current_point, discarded = next(self.step_generator)
                self.plot_step(hull, current_point, discarded, self.step_count)
                self.step_count += 1
                self.root.after(delay_ms, step)
            except StopIteration:
                messagebox.showinfo("Concluído", "Passo a passo finalizado!")

        step()


'''def create_sample_obj_file():
    """Cria um arquivo .obj de exemplo"""
    sample_points = [
        (0, 3), (1, 1), (2, 2), (4, 4), (0, 0), (1, 2), (3, 1), (3, 3),
        (2, 1), (1, 3), (4, 2), (2, 3), (1, 0), (3, 0), (4, 1), (0, 2),
        (2, 0), (0, 1), (4, 3), (3, 2)
    ]

    filename = "exemplo_pontos.obj"
    with open(filename, 'w') as f:
        f.write("# Arquivo de exemplo com pontos 2D\n")
        f.write("# Formato: v x y z (z será ignorado para pontos 2D)\n\n")
        for x, y in sample_points:
            f.write(f"v {x} {y} 0.0\n")

    print(f"Arquivo de exemplo criado: {filename}")
    return filename'''


if __name__ == "__main__":
    # Cria arquivo de exemplo
    #create_sample_obj_file()

    # Inicia a aplicação
    root = tk.Tk()
    app = ConvexHullGUI(root)
    root.mainloop()