# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sys
from pathlib import Path
from threading import Thread
from typing import List, Tuple, Dict

import cv2
import wx
from keras.models import Model
from wx import wxEVT_LEFT_DOWN, wxEVT_LEFT_UP, wxEVT_RIGHT_UP

import ml_training
import ml_evaluate

modulus = 400
log_time: Dict[str, int] = {}


class RedirectText(object):
    def __init__(self, aWxTextCtrl: wx.TextCtrl):
        self.out = aWxTextCtrl

    def write(self, string):
        wx.CallAfter(self.out.WriteText, string)
        self.flush()

    def flush(self):
        if wx.GetApp() is not None:
            wx.CallAfter(self.out.Refresh)


class Train(Thread):
    def __init__(self, this, level_name: str, epochs: int = 20):
        super().__init__()
        self.this: MainForm = this
        self.epochs = epochs
        self.level_name = level_name

    def run(self) -> None:
        ml_training.epoch = self.epochs
        ml_training.do_it(self.level_name, log_time=log_time, log_verbose=True)
        wx.CallAfter(self.this.train_done)

    @staticmethod
    def get_epochs() -> int:
        return ml_training.epoch

    @staticmethod
    def get_level_names() -> List[str]:
        return list(ml_training.level.keys())


class MainForm(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "TessiLab's Basic ML bench")

        main_panel = wx.Panel(self, wx.ID_ANY)
        log = wx.TextCtrl(main_panel, wx.ID_ANY, size=(300, 100),
                          style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        info = wx.FontInfo(12.0)
        info.Family(wx.FONTFAMILY_TELETYPE)
        info.Weight(wx.FONTWEIGHT_NORMAL)
        log.SetFont(wx.Font(info))

        south_panel = wx.Panel(main_panel, wx.ID_ANY)
        self.btn = wx.Button(south_panel, wx.ID_ANY, 'Launch training !')
        self.Bind(wx.EVT_BUTTON, self.onButton, self.btn)
        self.play_btn = wx.Button(south_panel, wx.ID_ANY, 'Play with it !')
        self.Bind(wx.EVT_BUTTON, self.onButtonPlay, self.play_btn)

        lbl = wx.StaticText(south_panel, wx.ID_ANY, 'Epochs')
        self.epochs = wx.TextCtrl(south_panel, wx.ID_ANY, f'{Train.get_epochs()}')
        self.radios: List[wx.RadioButton] = []

        # Add widgets to a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(log, 1, wx.ALL | wx.EXPAND, 5)
        sizer.Add(south_panel, 0, wx.ALL | wx.CENTER, 5)

        out_net_panel = wx.Panel(south_panel, wx.ID_ANY)
        out_net_sizer = wx.BoxSizer(wx.VERTICAL)
        # net_panel = wx.StaticBox(out_net_panel, wx.ID_ANY, 'Neural network size')
        # box = wx.RadioBox(south_panel, wx.ID_ANY, 'Neural network size')
        # net_sizer.Add(box)
        # net_sizer = wx.StaticBoxSizer(wx.VERTICAL, net_panel)

        net_panel = out_net_panel
        net_sizer = out_net_sizer
        text = wx.StaticText(net_panel, wx.ID_ANY, 'Neural network size   ')
        font: wx.Font = text.GetFont() # wx.Font(18, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        font.SetStyle(wx.FONTSTYLE_ITALIC)
        text.SetFont(font)
        net_sizer.Add(text)

        for label in Train.get_level_names():
            radio = wx.RadioButton(net_panel, wx.ID_ANY, label, name=label)
            net_sizer.Add(radio)
            self.radios.append(radio)
        net_panel.SetSizer(net_sizer)
        # out_net_sizer.Add(net_panel)
        out_net_panel.SetSizer(out_net_sizer)
        self.radios[0].SetValue(True)

        south_sizer = wx.BoxSizer(wx.HORIZONTAL)
        south_sizer.Add(out_net_panel) #, 0, wx.ALL, wx.LEFT, 5)
        south_sizer.Add(lbl)
        south_sizer.Add(self.epochs)
        south_sizer.Add(self.btn)
        south_sizer.Add(self.play_btn)
        south_panel.SetSizer(south_sizer)

        main_panel.SetSizer(sizer)

        # redirect text here
        redir = RedirectText(log)
        sys.stdout = redir
        # sys.stderr = redir
        self.train: Train = None

        self.Maximize(True)
        print("Click a button below.")
        wx.CallLater(1000, print, "Waiting...")

    def check_model_and_enable_ui(self):
        auto_path = Path("./auto")
        if auto_path.exists() and auto_path.is_dir():
            wx.CallAfter(self.play_btn.Enable, True)
        else:
            wx.CallAfter(self.play_btn.Enable, False)

    def train_done(self):
        self.btn.Enable(True)
        self.check_model_and_enable_ui()
        print()
        print('************')
        print('**  DONE  **')
        print('************')
        for k, v in log_time.items():
            print(f"{k}: {v} ms")
        print()
        print("Training is over. You may now play with the model: click the button bellow.")
        print("Left click draws a shape (digit).")
        print("Right click deletes the last segment.")
        print("The result is diplayed in the window title.")

    def onButton(self, event):
        global frame
        self.btn.Enable(False)
        try:
            ep = int(self.epochs.GetValue())
            name = 'basic'
            for r in self.radios:
                if r.GetValue():
                    name = r.GetName()
                    break
            print(f"Will train with neural network size {name}")
            self.train = Train(self, name, ep)
            self.train.start()
        except:
            self.btn.Enable(True)
            self.epochs.SetFocus()

    def onButtonPlay(self, event):
        global frame
        self.play_btn.Enable(False)
        frame = DrawPanel()
        frame.Show()
        self.Show(False)


class DrawPanel(wx.Frame):

    button_state: int
    points: List[List[Tuple[int, int]]]
    model: Model

    """Draw to a panel."""

    def __init__(self):
        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY, title="Draw a digit on Panel", size=(modulus, modulus + 28))
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.onMouseMove)

        self.button_state = 0
        self.points = []
        self.model = ml_evaluate.load_model('./auto')
        self.Bind(wx.EVT_CLOSE, self.close)

    def close(self, event):
        print("Closing")
        exit(0)

    def onMouseMove(self, event: wx.MouseEvent):
        if event.GetButton() == wx.MOUSE_BTN_LEFT:
            if event.GetEventType() == wxEVT_LEFT_UP:
                self.button_state = 0
                self.predict()
            elif event.GetEventType() == wxEVT_LEFT_DOWN:
                self.button_state = 1
                self.points.append([])
                self.points[-1].append((event.GetX(), event.GetY()))
            else:
                print(f"Event type {event.GetEventType()} <> {wxEVT_LEFT_DOWN} <> {wxEVT_LEFT_UP}")
        elif event.GetEventType() == wxEVT_RIGHT_UP and self.button_state == 0:
            if self.points:
                self.points.pop()
            if not self.points:
                self.SetTitle("Draw a digit on Panel")
            else:
                self.predict()

        elif self.button_state == 1:
            self.points[-1].append((event.GetX(), event.GetY()))
        self.Refresh()

    def OnPaint(self, event=None) -> wx.PaintDC:
        # dc.DrawLine(0,0, 100, 100)
        dc = wx.PaintDC(self)
        dc.SetBrush(wx.Brush("BLACK", wx.SOLID))
        size = self.GetSize()
        width = size.width
        dc.DrawRectangle(0, 0, width, size.height)
        for points in self.points:
            ep = int(width / 12)
            dc.SetPen(wx.Pen(wx.WHITE, ep))
            previous = None
            for point in points:
                if previous is None:
                    previous = point
                    continue
                else:
                    dc.DrawLine(previous[0], previous[1], point[0], point[1])
                    previous = point
        return dc

    def predict(self):
        image = self.paint_to_image()
        # th, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite("28.png", image)
        digit = ml_evaluate.predict(self.model, image)
        self.SetTitle(f"You drew a {digit} on Panel")

    def paint_to_image(self):
        dcSource = self.OnPaint()
        size = dcSource.Size

        # Create a Bitmap that will later on hold the screenshot image
        # Note that the Bitmap must have a size big enough to hold the screenshot
        # -1 means using the current default colour depth
        bmp = wx.EmptyBitmap(size.width, size.height)

        # Create a memory DC that will be used for actually taking the screenshot
        memDC = wx.MemoryDC()

        # Tell the memory DC to use our Bitmap
        # all drawing action on the memory DC will go to the Bitmap now
        memDC.SelectObject(bmp)

        # Blit (in this case copy) the actual screen on the memory DC
        # and thus the Bitmap
        memDC.Blit(0,  # Copy to this X coordinate
                   0,  # Copy to this Y coordinate
                   size.width,  # Copy this width
                   size.height,  # Copy this height
                   dcSource,  # From where do we copy?
                   0,  # What's the X offset in the original DC?
                   0  # What's the Y offset in the original DC?
                   )

        # Select the Bitmap out of the memory DC by selecting a new
        # uninitialized Bitmap
        memDC.SelectObject(wx.NullBitmap)

        img = bmp.ConvertToImage()
        img.SaveFile('saved.png', wx.BITMAP_TYPE_PNG)
        return cv2.imread("saved.png", cv2.IMREAD_GRAYSCALE)


def run(app: wx.App):
    frame = MainForm()
    frame.Show()

    app.MainLoop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = wx.App(False)
    run(app)
