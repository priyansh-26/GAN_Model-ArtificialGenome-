if self.__apply_dp:
            #     privacy_engine = PrivacyEngine()

            #     if self.__sigma == -1:
            #         # make_private_with_epsilon
            #         print('------> make_private_with_epsilon')
            #         self.__netD, d_optimizer, self.__train_loader\
            #             = privacy_engine.make_private_with_epsilon(
            #                 module=self.__netD,
            #                 optimizer=d_optimizer,
            #                 data_loader=self.__train_loader,
            #                 target_epsilon=self.__eps,
            #                 target_delta=self.__delta,
            #                 epochs=self.__epochs,
            #                 max_grad_norm=self.__max_per_sample_grad_norm,
            #             )
            #     else:
            #         # make_private
            #         print(f'------> make_private with sigma={self.__sigma}')
            #         self.__netD, d_optimizer, self.__train_loader\
            #             = privacy_engine.make_private(
            #                 module=self.__netD,
            #                 optimizer=d_optimizer,
            #                 data_loader=self.__train_loader,
            #                 noise_multiplier=self.__sigma,
            #                 max_grad_norm=self.__max_per_sample_grad_norm,
            #                 poisson_sampling=self.__use_poisson_sampling
            #             )