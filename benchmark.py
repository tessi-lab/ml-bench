import wx
import wx.adv


class Benchmark(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "Basic ML bench")
    pass


if __name__ == "__main__":
    app = wx.App(False)
    bitmap = wx.Bitmap('./assets/logo-tessilab.png', wx.BITMAP_TYPE_PNG)
    splash = wx.adv.SplashScreen(bitmap, wx.adv.SPLASH_CENTRE_ON_SCREEN | wx.adv.SPLASH_TIMEOUT,
                                 6000, None, -1, wx.DefaultPosition, wx.DefaultSize,
                                 wx.BORDER_SIMPLE | wx.STAY_ON_TOP)

    wx.Yield()
    import main
    main.run(app)
