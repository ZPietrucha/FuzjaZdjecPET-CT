import concurrent.futures
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pydicom
import os
import cv2
import pywt

postprocess_CT = []
postprocess_PET = []
to_show = []
images = []


def import_file():
    """
    Funkcja Importuje obrazy DICOMowskie z wybranego folderu i zwraca listę infomracje o pikselach obrazu.
    Używa tkintera aby otworzyć okno dialogowe do wyboru folderu i następnie przy pomocy pydicom czyta wszystkie
    pliki z folderu i dodaje ich macierze pikseli do list Jeśli został zgłoszony wyjątek wyświetla się okienko błędu
    i jeśli to potrzebne są zerowane listy postporcess_CT i postpocess_PET.
    :return:lista macierzy obrazów DICOMowskich
    """
    list1 = []
    try:
        filepath = tk.filedialog.askdirectory()
        lista = os.listdir(filepath)
        for each in lista:
            dcm = pydicom.dcmread(filepath + "/" + each, force=True)
            if 'PixelData' in dcm:
                imageData = dcm.pixel_array
                list1.append(imageData)
    except Exception:
        tk.messagebox.showerror("Plik", "Nie wybrano foldera")
        try:
            if len(postprocess_CT[0]) < 10:
                postprocess_CT.clear()
            if len(postprocess_PET[0]) < 10:
                postprocess_PET.clear()
        except IndexError:
            postprocess_CT.clear()
            postprocess_PET.clear()
    return list1


def dwt2RGB(matrix):
    """
    Funkcja oblicza 2D trasnforamtę falkową zadanego zdjęica przy użyciu falki haara.
    :param matrix: macierz pikseli zdjec
    :return: (tupel) współczynniki transformaty falkowej
    """
    coefficients = pywt.dwt2(matrix, "haar", mode="periodization")
    LL, (LH, HL, HH) = coefficients
    coeff2 = LL, (LH, HL, HH)
    return coeff2


def PET_processing(image):
    """
    Funkcja przetwarza obraz PET tak, żeby miał odpowiednią wielkość oraz nadaje mu barwy, po czym liczy jego
    trasnformatę falkową dla każdego z kanałów.
    :param image: zdjęcie przedstawiajace wynik PET do obróbki
    :return: lista współczynników 2D transformaty falkowej dla każdego kanału RGB
    """
    normalized = np.zeros((512,512))
    normalized = cv2.normalize(image,normalized,0,255,cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    resized = cv2.resize(normalized, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    colored = cv2.applyColorMap(resized, 13)
    R = colored[:, :, 2]
    G = colored[:, :, 1]
    B = colored[:, :, 0]
    RGB = [R, G, B]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        coefficients = list(executor.map(dwt2RGB, RGB))
    return coefficients


def CT_processing(image):
    """
    Funkcja przetwarza obraz CT normalizując jego wartości do 8bitowych i nadając mu odpowiednie kolory, po czym
    liczy jego 2D trasnformatę falkową dla każego z kanałów
    :param image: obraz CT przeznaczony do obróbki
    :return: lista współczynników 2D transformaty falkowej dla każdego kanału RGB
    """
    normalized = np.zeros((512,512))
    normalized = cv2.normalize(image,normalized,0,255,cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
    R = colored[:, :, 2]
    G = colored[:, :, 1]
    B = colored[:, :, 0]
    RGB = [R, G, B]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        coefficients = list(executor.map(dwt2RGB, RGB))
    return coefficients


def CT_output():
    """
    Funkcja wczytuje zdjęcia i wykonuje funcję CT_processing na każdym ze wgranych zdjęć z folderu
    :return:lista współczynników wszystkich zdjęć CT załadowanych do programu
    """
    if len(postprocess_CT) == 0 or len(postprocess_CT[0]) == 1:
        CT_images = import_file()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            postprocess_CT.append(list(executor.map(CT_processing, CT_images)))
        if (len(postprocess_CT[0])) > 1:
            tk.messagebox.showinfo("CT", "Załadowano zdjęcia CT")
    else:
        tk.messagebox.showerror("CT", "Zdjecia już zostały załadowane!")



def PET_output():
    """
    Funkcja wczytuje zdjęcia i wykonuje funcję PET_processing na każdym ze wgranych zdjęć z folderu
    :return:lista współczynników wszystkich zdjęć PET załadowanych do programu
    """
    if (len(postprocess_PET)) == 0 or len(postprocess_PET[0]) == 1:
        PET_images = import_file()
        PET_images.reverse()
        PET_images = [x for i, x in enumerate(PET_images) if i % 5 != 4]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            postprocess_PET.append(list(executor.map(PET_processing, PET_images)))
        if(len(postprocess_PET[0])) > 1:
            tk.messagebox.showinfo("PET", "Załadowano zdjęcia PET")
    else:
        tk.messagebox.showerror("PET", "Zdjecia już zostały załadowane!")


def fusion(PET_image, CT_image):
    """
    Funkcja odpowiedzialna za fuzję dwóch zdjęć, PET i CT, według kryterium MAX-AVR
    :param PET_image:współczynniki transformaty falkowej obrazu PET przeznaczonego do fuzji
    :param CT_image:współczynniki transformaty falkowej obrazu CT przeznaczonego do fuzji
    :return:3D array reprezentujący zdjęcie po fuzji
    """
    #weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    weight = 0.7
    LLR = np.maximum(PET_image[0][0], CT_image[0][0])
    LHR = CT_image[0][1][0] * weight + PET_image[0][1][0] * (1 - weight)
    HLR = CT_image[0][1][1] * weight + PET_image[0][1][1] * (1 - weight)
    HHR = CT_image[0][1][2] * weight + PET_image[0][1][2] * (1 - weight)

    LLG = np.maximum(PET_image[1][0], CT_image[1][0])
    LHG = CT_image[1][1][0] * weight + PET_image[1][1][0] * (1 - weight)
    HLG = CT_image[1][1][1] * weight + PET_image[1][1][1] * (1 - weight)
    HHG = CT_image[1][1][2] * weight + PET_image[1][1][2] * (1 - weight)

    LLB = np.maximum(PET_image[2][0], CT_image[2][0])
    LHB = CT_image[2][1][0] * weight + PET_image[2][1][0] * (1 - weight)
    HLB = CT_image[2][1][1] * weight + PET_image[2][1][1] * (1 - weight)
    HHB = CT_image[2][1][2] * weight + PET_image[2][1][2] * (1 - weight)
    Rcoeff = LLR, (LHR, HLR, HHR)
    Gcoeff = LLG, (LHG, HLG, HHG)
    Bcoeff = LLB, (LHB, HLB, HHB)
    R = pywt.idwt2(Rcoeff, "haar", mode="periodization")
    G = pywt.idwt2(Gcoeff, "haar", mode="periodization")
    B = pywt.idwt2(Bcoeff, "haar", mode="periodization")
    fused = np.dstack((R, G, B))
    return fused


def PET_CT_fusion():
    """
    Funkcja łącząca odpowiadające sobie obrazy PET i CT przy pomocy funkcji fusion. Obrazy po fuzji są przekształcane
    na Image możliwy do wyświetlenia na obiekcie Canvas z tkinter.
    :return: lista obrazów po fuzji
    """
    if len(postprocess_PET) == 0 or len(postprocess_CT) == 0:
        tk.messagebox.showerror(title="Błąd", message="Brak zdjęć")
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
                to_show = list(executor.map(fusion, postprocess_PET[0], postprocess_CT[0]))
        for each in to_show:
            img = ImageTk.PhotoImage(image=Image.fromarray(np.uint8(each), mode="RGB"))
            images.append(img)
        tk.messagebox.showinfo("Fuzja", "Fuzja ukończona pomyślnie, przesuń suwakiem po lewo, żeby wyświetlić zdjęcia")


def clear_data(e):
    """
    Funkcja usuwająca wszelkie dane progamu po wciśnięciu klawisza DELETE
    :param e: klawisz DELETE
    """
    postprocess_CT.clear()
    postprocess_PET.clear()
    to_show.clear()
    images.clear()
    tk.messagebox.showinfo("Usuwanie", "Dane zostały usunięte")

def create_image(number):
    """
    Funkcja wyświetlająca zdjęcia z pamięci programu na obiekt tkinkter.Canvas.
    W przypadku braku zdjęć do wyświetlenia wyskakuje odpowiedni komunikat.
    :param number: indykator zdjęcia ustawiany przez obiekt klasy tkinkter.Scale
    """
    try:
        number = int(number)
        canvas.create_image(0, 0, anchor=tk.NW, image=images[number])
    except IndexError:
        tk.messagebox.showerror(title="Fuzja", message="Brak zdjęć")
        change_image_scale.set(0)


def change_image(val):
    """
    Funkcja odpowiadająca za zmianę kolejnych zdjęć przy pomocy obiektu klasy tkinkter.Scale
    :param val: wartość suwaka
    """
    create_image(change_image_scale.get())


if __name__ == "__main__":
    window = tk.Tk()
    window.title("Fuzja PET i CT")
    window.geometry("700x700")
    # buttons
    import_CT = tk.Button(window, text="CT", command=CT_output)
    import_PET = tk.Button(window, text="PET", command=PET_output)
    perform_fusion = tk.Button(window, text="Fusion", command=PET_CT_fusion)
    destroy_data = tk.Button(window, text="Delete data", command=clear_data)
    # scales
    change_image_scale = tk.Scale(window, from_=0, to=203, resolution=1,length=512, command=create_image)
    #change_weight = tk.Scale(window, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL)
    # canvas
    canvas = tk.Canvas(window, width=512, height=512)
    # pack
    import_CT.pack()
    import_PET.pack()
    perform_fusion.pack()
    change_image_scale.pack(side=tk.LEFT)
    #change_weight.pack(side=tk.BOTTOM)
    canvas.pack()
    # bind
    window.bind("<Delete>", clear_data)
    window.mainloop()
