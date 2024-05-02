from conditional_diffusion.modules import UNet


class UNet_conditional(UNet): #modified version of conditional_diffusion.UNet_conditional
    def __init__(self, c_in=3, c_out=3, time_dim=512, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        #time_dim dictates what embedding size you can use

    def forward(self, x, t, embedding):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        t += embedding

        return self.unet_forwad(x, t)