from pathlib import Path

import torch
from xca.ml.torch.cnn import EnsembleCNN
from xca.ml.torch.training import JointVAEClassifierModule, dynamic_training
from xca.ml.torch.vae import VAE, CNNDecoder, CNNEncoder

# BEGIN XRD PARAMETERS #
param_dict = {
    "wavelength": 0.1665,
    "noise_std": 7e-3,
    "instrument_radius": 1000.0,
    "theta_m": 0.0,
    "tth_min": 0.1,
    "tth_max": 12.0,
    "n_datapoints": 3000,
}
kwargs = {
    "bkg_2": (-1e-4, 1e-4),
    "bkg_1": (-1e-4, 1e-4),
    "bkg_0": (0, 1e-3),
    "march_range": (0.8, 1.0),
    "isotropic_expansion": (-0.05, 0.05),
}
shape_limit = 1e-2

cif_paths = list((Path(__file__).parent / "phases").glob("*.cif"))


def joint_vae_class_main(checkpoint=None):
    if checkpoint is None:
        cnn = EnsembleCNN(
            ensemble_size=25,
            filters=[8, 8, 4],
            kernel_sizes=[5, 5, 5],
            strides=[2, 2, 2],
            pool_sizes=[1, 1, 1],
            n_classes=10,
            ReLU_alpha=0.2,
            dense_dropout=0.4,
        )
        encoder = CNNEncoder(
            input_length=3000,
            latent_dim=2,
            filters=(8, 4),
            kernel_sizes=(5, 5),
            strides=(2, 1),
            pool_sizes=(2, 2),
        )
        decoder = CNNDecoder.from_encoder(encoder)

        vae = VAE(encoder, decoder)
        pl_module = JointVAEClassifierModule(cnn, vae, classification_lr=1e-4, vae_lr=1e-4, kl_weight=1e-2)
    else:
        ckpt = torch.load(checkpoint)
        cnn = EnsembleCNN(**ckpt["hyper_parameters"]["classifier_hparams"])
        encoder = CNNEncoder(**ckpt["hyper_parameters"]["encoder_hparams"])
        decoder = CNNDecoder(**ckpt["hyper_parameters"]["decoder_hparams"])
        vae = VAE(encoder, decoder)
        pl_module = JointVAEClassifierModule.load_from_checkpoint(
            checkpoint, classification_model=cnn, vae_model=vae
        )

    metrics = dynamic_training(
        pl_module,
        gpus=[1],
        max_epochs=500,
        batch_size=24,
        num_workers=24,
        prefetch_factor=4,
        batch_per_train_epoch=500,
        batch_per_val_epoch=1,
        cif_paths=cif_paths,
        param_dict=param_dict,
        shape_limit=shape_limit,
        **kwargs,
    )
    return pl_module, metrics


if __name__ == "__main__":
    joint_vae_class_main()
