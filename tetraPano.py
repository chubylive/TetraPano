import octahedron as oct
import cube as cub
import iconsohedron as ico
import tetrahedron as tet
import dodecahedron as dod
import PySimpleGUI as sg
BAR_MAX = 1000

currentProcGeod = ""
sg.theme('DarkAmber')
layout = [[sg.Text('Select which geodesic to map.')],               #0
          [sg.Checkbox('CUBE')],                                    #1
          [sg.Checkbox('ICOS')],                                    #2
          [sg.Checkbox('DODEC')],                                   #3
          [sg.Checkbox('OCTA')],                                    #4
          [sg.Checkbox('TETRA')],                                   #5
          [sg.Text("Choose a 2:1 image: "), sg.FileBrowse(key="-IN-")],       #6
          [sg.Text("Save location:      "), sg.FolderBrowse(key="-OUT-")],     #7
          [sg.Submit(), sg.Cancel()]                                #8
          ]
layout_1 = [[sg.Text('Progress Bar for ' + currentProcGeod )],
          [sg.ProgressBar(BAR_MAX, orientation='h', size=(20,20), key='-PROG-')],
          [sg.Cancel()]]

window = sg.Window('TETRA PANO', layout, font=("Helvetica", 12))      

event, values = window.read()
# print(values["-IN-"], values["-OUT-"])
if(event == 'Submit'):
    if (values[0] == 1):
        print ("mapping cube")
        cub.makeNet(values["-OUT-"], values["-IN-"],4)
    if (values[1] == 1):
        print ("mapping icosohedron")
        ico.makeNet(values["-OUT-"], values["-IN-"],4)
    if (values[2] == 1):
        print ("mapping dedecadron")
        dod.makeNet(values["-OUT-"], values["-IN-"],4)
    if (values[3] == 1):
        print ("mapping octahedron")
        oct.makeNet(values["-OUT-"], values["-IN-"],4)
    if (values[4] == 1):
        print ("mapping tetrahedron")
        tet.makeNet(values["-OUT-"], values["-IN-"],4)