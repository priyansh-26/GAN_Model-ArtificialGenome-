from modules import meminf, imputation, training
from modules.training.gan.train_GAN import train

class PrivacyProtectedArtificialGenomes:
    """
    A class that encompasses everything from model training to evaluation.
    """
    def __init__(self, args):
        self.__args = args

        # define variables from args
        self.__train = args.train
        self.__regenerate = args.regenerate
        self.__imputation = args.imputation
        self.__wb_attack = args.wb_attack


    def _train_model(self):
        """
        create the model.
        """
        _, _ = train(self.args)


    def _regenerate(self):
        """
        regenerate haplotypes from trained models.
        """
        training.gan.functions.regenerate_artificial_genomes(self.__args)


    def _exec_imputation(self):
        """
        Execute genotype imputation using IMPUTE2
        """
        im2 = imputation.Imputation(
            args=self.args
        )
        im2.main()


    def _membership_inference(self):
        """
        Execude membership inference attacks
        """
        mia = meminf.MembershipInferenceAttack(args=self.args)
        mia.exec_attack()


    def main(self) -> None:
        # model's training
        if self.__train:
            self._train_model()

        # regenerate haplotypes
        if self.__regenerate:
            self._regenerate()

        # evaluation (genotype imputation)
        if self.__imputation:
            self._exec_imputation()

        # evaluation (membership inference attacks)
        if self.__wb_attack:
            self._membership_inference()

    @property
    def args(self):
        return self.__args