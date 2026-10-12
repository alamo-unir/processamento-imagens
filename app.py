import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

import processamento as proc

class SistemaProcessamentoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Processamento de Imagens - UNIR")
        self.root.geometry("1100x740")
        self.root.config(bg="#2b2b2b")
        
        self.caminho_imagem = None
        self.img_original = None
        self.img_processada = None

        # --- PAINEL DE CONTROLE (Esquerda) ---
        frame_controle = tk.Frame(root, bg="#3c3f41", width=320)
        frame_controle.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        lbl_titulo = tk.Label(frame_controle, text="Menu de Operações", bg="#3c3f41", fg="white", font=("Arial", 12, "bold"))
        lbl_titulo.pack(pady=10)
        
        # Botões de Arquivo
        btn_carregar = tk.Button(frame_controle, text="📁 Carregar Imagem 1", command=self.carregar_imagem, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT)
        btn_carregar.pack(fill=tk.X, padx=10, pady=2)
        
        btn_salvar = tk.Button(frame_controle, text="💾 Salvar Processada", command=self.salvar_imagem, bg="#FF9800", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT)
        btn_salvar.pack(fill=tk.X, padx=10, pady=5)
        
        # Categorias de Operações
        self.criar_secao(frame_controle, "Operações Algébricas", [
            ("Dissolve Cruzado Uniforme", self.executar_dissolve_uniforme_gui),
            ("Dissolve Não-Uniforme", self.executar_dissolve_nao_uniforme)
        ])

        self.criar_secao(frame_controle, "Transformações de Intensidade", [
            ("Negativo", lambda: self.executar_simples(proc.negative)),
            ("Logaritmo", lambda: self.executar_simples(proc.log)),
            ("Alargamento de Contraste", lambda: self.executar_simples(proc.contrast_stretch)),
            ("Potência (Gamma)", self.executar_gamma),
            ("Limiarização (Threshold)", self.executar_threshold)
        ])
        
        self.criar_secao(frame_controle, "Histograma", [
            ("Equalização", lambda: self.executar_simples(proc.histogram_equalization)),
            ("Expansão", lambda: self.executar_simples(proc.histogram_expansion))
        ])
        
        self.criar_secao(frame_controle, "Filtros Espaciais", [
            ("Filtro da Média", lambda: self.executar_filtro_espacial(proc.mean)),
            ("Filtro da Mediana", lambda: self.executar_filtro_espacial(proc.median)),
            ("Contraste Adaptativo", self.executar_contraste_adaptativo)
        ])
        
        self.criar_secao(frame_controle, "Aguçamento & Bordas", [
            ("Aguçamento Laplaciano", lambda: self.executar_simples(proc.laplacian_sharpening)),
            ("High-Boost", self.executar_highboost),
            ("Bordas: Laplaciano", lambda: self.executar_simples(proc.laplacian_edges)),
            ("Bordas: Roberts", self.executar_roberts),
            ("Bordas: Sobel", self.executar_sobel),
            ("Bordas: Kirsch", lambda: self.executar_simples(proc.kirsch))
        ])

        self.criar_secao(frame_controle, "Avançado", [
            ("Convolução Genérica (Customizada)", self.executar_convolucao_generica)
        ])

        # --- PAINEL DE VISUALIZAÇÃO (Direita) ---
        frame_visualizacao = tk.Frame(root, bg="#2b2b2b")
        frame_visualizacao.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Frame Imagem Original
        frame_orig = tk.Frame(frame_visualizacao, bg="#3c3f41")
        frame_orig.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        tk.Label(frame_orig, text="Imagem Original (1)", bg="#3c3f41", fg="white", font=("Arial", 11, "bold")).pack(pady=5)
        self.lbl_orig = tk.Label(frame_orig, text="Sem imagem", bg="#2b2b2b", fg="gray")
        self.lbl_orig.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Frame Imagem Processada
        frame_proc = tk.Frame(frame_visualizacao, bg="#3c3f41")
        frame_proc.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)
        tk.Label(frame_proc, text="Imagem Processada", bg="#3c3f41", fg="white", font=("Arial", 11, "bold")).pack(pady=5)
        self.lbl_proc = tk.Label(frame_proc, text="Aguardando...", bg="#2b2b2b", fg="gray")
        self.lbl_proc.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

    def criar_secao(self, parent, titulo, botoes):
        tk.Label(parent, text=titulo, bg="#3c3f41", fg="#00bcd4", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        for texto, comando in botoes:
            btn = tk.Button(parent, text=texto, command=comando, bg="#505354", fg="white", anchor="w", relief=tk.FLAT, font=("Arial", 9))
            btn.pack(fill=tk.X, padx=15, pady=1)

    def ler_imagem_unicode(self, caminho):
        """Lê imagem suportando caminhos com acentos no Windows."""
        dados = np.fromfile(caminho, dtype=np.uint8)
        return cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)

    def salvar_imagem_unicode(self, caminho, imagem):
        """Salva imagem suportando caminhos com acentos no Windows."""
        extensao = os.path.splitext(caminho)[1]
        sucesso, buffer = cv2.imencode(extensao, imagem)
        if sucesso:
            buffer.tofile(caminho)

    def carregar_imagem(self):
        file_path = filedialog.askopenfilename(parent=self.root, filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.caminho_imagem = file_path
            self.img_original = self.ler_imagem_unicode(file_path)
            if self.img_original is not None:
                self.exibir_na_tela(self.img_original, self.lbl_orig)
                self.lbl_proc.config(image="", text="Aguardando...")
                self.img_processada = None

    def salvar_imagem(self):
        if self.img_processada is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem processada para salvar!", parent=self.root)
            return
        file_path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".png", filetypes=[("PNG file", "*.png"), ("JPEG file", "*.jpg")])
        if file_path:
            self.salvar_imagem_unicode(file_path, self.img_processada)
            messagebox.showinfo("Sucesso", f"Imagem salva com sucesso em:\n{file_path}", parent=self.root)

    def exibir_na_tela(self, img_cv2, label_widget):
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((420, 420))
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

    # --- MÉTODOS DE EXECUÇÃO (Com parent=self.root para garantir foco) ---
    def executar_simples(self, funcao):
        if self.img_original is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        self.img_processada = funcao(self.img_original)
        self.exibir_na_tela(self.img_processada, self.lbl_proc)

    def executar_gamma(self):
        if self.img_original is None: 
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        p = simpledialog.askfloat("Gamma", "Digite o valor de p (gamma):", initialvalue=1.5, parent=self.root)
        if p is not None:
            self.img_processada = proc.gamma(self.img_original, p)
            self.exibir_na_tela(self.img_processada, self.lbl_proc)

    def executar_threshold(self):
        if self.img_original is None: 
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        t = simpledialog.askinteger("Limiarização", "Digite o limiar (0-255):", initialvalue=127, parent=self.root)
        if t is not None:
            self.img_processada = proc.threshold(self.img_original, t)
            self.exibir_na_tela(self.img_processada, self.lbl_proc)

    def executar_filtro_espacial(self, funcao_filtro):
        if self.img_original is None: 
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        try:
            m = simpledialog.askinteger("Filtro Espacial", "Linhas do kernel (m - ÍMPAR):", initialvalue=3, parent=self.root)
            n = simpledialog.askinteger("Filtro Espacial", "Colunas do kernel (n - ÍMPAR):", initialvalue=3, parent=self.root)
            if m and n:
                if m % 2 == 0 or n % 2 == 0:
                    raise ValueError("As dimensões do kernel devem ser ímpares!")
                self.img_processada = funcao_filtro(self.img_original, m, n)
                self.exibir_na_tela(self.img_processada, self.lbl_proc)
        except ValueError as e:
            messagebox.showerror("Erro de Dimensão", str(e), parent=self.root)

    def executar_contraste_adaptativo(self):
        if self.img_original is None: 
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        n = simpledialog.askinteger("Contraste Adaptativo", "Tamanho da vizinhança n (ÍMPAR):", initialvalue=3, parent=self.root)
        if n is None: return
        if n % 2 == 0:
            messagebox.showerror("Erro", "O tamanho da vizinhança n deve ser ímpar!", parent=self.root)
            return
        c = simpledialog.askfloat("Contraste Adaptativo", "Constante c:", initialvalue=1.0, parent=self.root)
        if c is not None:
            self.img_processada = proc.adaptive_contrast(self.img_original, n, c)
            self.exibir_na_tela(self.img_processada, self.lbl_proc)

    def executar_highboost(self):
        if self.img_original is None: 
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        k = simpledialog.askfloat("High-Boost", "Digite o fator k:", initialvalue=1.0, parent=self.root)
        if k is not None:
            self.img_processada = proc.highboost_sharpening(self.img_original, k)
            self.exibir_na_tela(self.img_processada, self.lbl_proc)

    def executar_roberts(self):
        if self.img_original is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        _, _, magnitude = proc.roberts(self.img_original)
        self.img_processada = magnitude
        self.exibir_na_tela(self.img_processada, self.lbl_proc)

    def executar_sobel(self):
        if self.img_original is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        _, _, magnitude = proc.sobel(self.img_original)
        self.img_processada = magnitude
        self.exibir_na_tela(self.img_processada, self.lbl_proc)

    def executar_dissolve_uniforme_gui(self):
        if self.img_original is None:
            messagebox.showwarning("Aviso", "Carregue a Imagem 1 primeiro!", parent=self.root)
            return
            
        file_path = filedialog.askopenfilename(parent=self.root, title="Selecione a Imagem 2 para o Dissolve", filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path: return
        img2 = cv2.imread(file_path, 0)
        
        if img2.shape != self.img_original.shape:
            img2 = cv2.resize(img2, (self.img_original.shape[1], self.img_original.shape[0]))

        janela_dissolve = tk.Toplevel(self.root)
        janela_dissolve.title("Dissolve Cruzado Uniforme Interativo")
        janela_dissolve.geometry("500x550")
        janela_dissolve.config(bg="#3c3f41")
        
        # Força a janela a ficar por cima e travar o foco (Modal)
        janela_dissolve.transient(self.root)
        janela_dissolve.grab_set()
        
        tk.Label(janela_dissolve, text="Ajuste o fator t (0.0 = Img 1 | 1.0 = Img 2)", bg="#3c3f41", fg="white", font=("Arial", 11, "bold")).pack(pady=10)
        
        lbl_preview = tk.Label(janela_dissolve, bg="#2b2b2b")
        lbl_preview.pack(pady=10)
        
        def atualizar_preview(val):
            t = float(val)
            resultado = proc.uniform_cross_dissolve(self.img_original, img2, t)
            self.img_processada = resultado 
            
            img_rgb = cv2.cvtColor(resultado, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((350, 350))
            img_tk = ImageTk.PhotoImage(img_pil)
            lbl_preview.config(image=img_tk)
            lbl_preview.image = img_tk
            self.exibir_na_tela(resultado, self.lbl_proc)

        slider = ttk.Scale(janela_dissolve, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=atualizar_preview)
        slider.set(0.5)
        slider.pack(fill=tk.X, padx=30, pady=10)
        
        atualizar_preview(0.5)
        
        tk.Button(janela_dissolve, text="Concluir / Fechar", command=janela_dissolve.destroy, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT).pack(pady=10)

    def executar_dissolve_nao_uniforme(self):
        if self.img_original is None:
            messagebox.showwarning("Aviso", "Carregue a Imagem 1 primeiro!", parent=self.root)
            return
            
        file_path = filedialog.askopenfilename(parent=self.root, title="Selecione a Imagem 2 para o Dissolve Não-Uniforme", filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path: return
        img2 = cv2.imread(file_path, 0)
        
        if img2.shape != self.img_original.shape:
            img2 = cv2.resize(img2, (self.img_original.shape[1], self.img_original.shape[0]))
            
        self.img_processada = proc.non_uniform_cross_dissolve(self.img_original, img2)
        self.exibir_na_tela(self.img_processada, self.lbl_proc)
        messagebox.showinfo("Sucesso", "Dissolve Não-Uniforme aplicado com sucesso!", parent=self.root)

    def executar_convolucao_generica(self):
        if self.img_original is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!", parent=self.root)
            return
        
        janela = tk.Toplevel(self.root)
        janela.title("Convolução Genérica Customizada")
        janela.geometry("400x380")
        janela.config(bg="#3c3f41")
        
        janela.transient(self.root)
        janela.grab_set()
        
        tk.Label(janela, text="Configuração do Kernel (Máscara h)", bg="#3c3f41", fg="white", font=("Arial", 11, "bold")).pack(pady=10)
        
        frame_dims = tk.Frame(janela, bg="#3c3f41")
        frame_dims.pack(pady=5)
        
        tk.Label(frame_dims, text="Linhas m (ÍMPAR):", bg="#3c3f41", fg="white").pack(side=tk.LEFT, padx=5)
        entry_m = tk.Entry(frame_dims, width=5)
        entry_m.insert(0, "3")
        entry_m.pack(side=tk.LEFT, padx=5)
        
        tk.Label(frame_dims, text="Colunas n (ÍMPAR):", bg="#3c3f41", fg="white").pack(side=tk.LEFT, padx=5)
        entry_n = tk.Entry(frame_dims, width=5)
        entry_n.insert(0, "3")
        entry_n.pack(side=tk.LEFT, padx=5)
        
        tk.Label(janela, text="Digite os valores linha por linha (espaçados):", bg="#3c3f41", fg="#cccccc", font=("Arial", 9)).pack(pady=(10, 2))
        
        text_kernel = tk.Text(janela, height=5, width=30)
        text_kernel.insert(tk.END, "0 1 0\n1 -4 1\n0 1 0")
        text_kernel.pack(pady=5)
        
        frame_offset = tk.Frame(janela, bg="#3c3f41")
        frame_offset.pack(pady=5)
        tk.Label(frame_offset, text="Offset:", bg="#3c3f41", fg="white").pack(side=tk.LEFT, padx=5)
        entry_offset = tk.Entry(frame_offset, width=10)
        entry_offset.insert(0, "0")
        entry_offset.pack(side=tk.LEFT, padx=5)
        
        def aplicar():
            try:
                m = int(entry_m.get())
                n = int(entry_n.get())
                
                if m % 2 == 0 or n % 2 == 0:
                    messagebox.showerror("Erro de Validação", "As dimensões m e n do kernel devem ser obrigatoriamente ÍMPARES (ex: 3x3, 5x5)!", parent=janela)
                    return
                
                offset = float(entry_offset.get())
                
                texto_linhas = text_kernel.get("1.0", tk.END).strip().split("\n")
                if len(texto_linhas) != m:
                    messagebox.showerror("Erro", f"O número de linhas digitadas difere de m={m}!", parent=janela)
                    return
                
                linhas = []
                for linha_str in texto_linhas:
                    valores = list(map(float, linha_str.split()))
                    if len(valores) != n:
                        messagebox.showerror("Erro", f"Cada linha deve conter exatamente {n} valores!", parent=janela)
                        return
                    linhas.append(valores)
                
                h = np.array(linhas, dtype=np.float64)
                
                self.img_processada = proc.convolution(self.img_original, h, offset=offset, mode="edge")
                self.exibir_na_tela(self.img_processada, self.lbl_proc)
                janela.destroy()
                messagebox.showinfo("Sucesso", "Convolução genérica aplicada com sucesso!", parent=self.root)
                
            except ValueError as ve:
                messagebox.showerror("Erro de Valor", f"Certifique-se de digitar números válidos:\n{str(ve)}", parent=janela)
            except Exception as e:
                messagebox.showerror("Erro", f"Ocorreu um erro:\n{str(e)}", parent=janela)

        tk.Button(janela, text="Aplicar Convolução", command=aplicar, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaProcessamentoApp(root)
    root.mainloop()