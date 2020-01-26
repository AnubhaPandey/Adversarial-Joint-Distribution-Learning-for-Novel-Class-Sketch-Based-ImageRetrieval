from layer import *
from mmd_loss import *

class Network:
    def __init__(self, x, y, is_training, batch_size):
        self.batch_size = batch_size
        #y -- video feature, x -- text embedding
        self.x = x
        self.y = y
        #1. q(x)p(y|x)
        z2 = tf.random_normal([self.batch_size,500],0, 1)
        self.gen_y = self.generator1(tf.concat([self.x,z2],axis=-1) , is_training, reuse=tf.AUTO_REUSE)
        self.D1_real = self.discriminator1(tf.concat([self.x,self.y],axis=-1), reuse=False)
        self.D1_fake = self.discriminator1(tf.concat([self.x,self.gen_y],axis=-1), reuse=True)

        #2. q(y)p(x|y)
        z1 = tf.random_normal([self.batch_size,500],0, 1)
        self.gen_x = self.generator2(tf.concat([self.y,z1],axis=-1) , is_training, reuse=tf.AUTO_REUSE)
        self.D2_real = self.discriminator2(tf.concat([self.x,self.y],axis=-1), reuse=False)
        self.D2_fake = self.discriminator2(tf.concat([self.gen_x,self.y],axis=-1), reuse=True)

        #3. p(x)p(y|x)
        y_zero = tf.zeros_like(self.y)
        latent_z1 = tf.random_normal([self.batch_size,500],0, 1)
        fake_x = self.generator2(tf.concat([y_zero,latent_z1], axis=-1), is_training, reuse=tf.AUTO_REUSE)
        z2_ = tf.random_normal([self.batch_size,500],0, 1)
        self.gen_fake_y = self.generator1(tf.concat([fake_x,z2_],axis=-1) , is_training, reuse=tf.AUTO_REUSE)
        self.D3_real = self.discriminator3(self.y, reuse=False)
        self.D3_fake = self.discriminator3(self.gen_fake_y, reuse=True)

        #4.p(y)p(x|y)
        x_zero = tf.zeros_like(self.x)
        latent_z2 = tf.random_normal([self.batch_size,500],0, 1)
        fake_y = self.generator1(tf.concat([x_zero,latent_z2], axis=-1), is_training, reuse=tf.AUTO_REUSE)
        z1_ = tf.random_normal([self.batch_size,500],0, 1)
        self.gen_fake_x = self.generator2(tf.concat([fake_y,z1_],axis=-1) , is_training, reuse=tf.AUTO_REUSE)
        self.D4_real = self.discriminator4(self.x, reuse=False)
        self.D4_fake = self.discriminator4(self.gen_fake_x, reuse=True)

        #5. cycle consistency
        #q(x)p(y|x)p(x|y)
        z3_ = tf.random_normal([self.batch_size,500],0, 1)
        self.gen_x1 = self.generator2(tf.concat([self.gen_y,z3_],axis=-1) , is_training, reuse=tf.AUTO_REUSE)
        self.D5_real = self.discriminator5(self.x, reuse=False)
        self.D5_fake = self.discriminator5(self.gen_x1, reuse=True)
        #q(y)p(x|y)p(y|x)
        z4_ = tf.random_normal([self.batch_size,500],0, 1)
        self.gen_y1 = self.generator1(tf.concat([self.gen_x,z4_],axis=-1) , is_training, reuse=tf.AUTO_REUSE)
        self.D6_real = self.discriminator6(self.y, reuse=False)
        self.D6_fake = self.discriminator6(self.gen_y1, reuse=True)


        self.g_loss = self.calc_g_loss()
        self.d_loss = self.calc_d_loss()
        self.gan_loss = self.g_loss + 0.0001*self.d_loss

        all_vars = tf.trainable_variables()
        self.gan_variables = [var for var in all_vars]
        #self.g_variables = [var for var in all_vars]
        self.g_variables = [var for var in all_vars if var.name.startswith('g')]
        self.d_variables = [var for var in all_vars if var.name.startswith('d')]


    def generator1(self, x, is_training, reuse):
        a = self.y
        with tf.variable_scope('g1', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc4'):
                x = full_connection_layer(x, a.shape[1])
        return x


    def generator2(self, x, is_training, reuse):
        a = self.x
        with tf.variable_scope('g2', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc4'):
                x = full_connection_layer(x, a.shape[1])
        return x


    def discriminator1(self, x, reuse):
        is_training = tf.constant(True)
        with tf.variable_scope('d1', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 128)
        return x


    def discriminator2(self, x, reuse):
        is_training = tf.constant(True)
        with tf.variable_scope('d2', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 128)
        return x


    def discriminator3(self, x, reuse):
        is_training = tf.constant(True)
        with tf.variable_scope('d3', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 128)
        return x


    def discriminator4(self, x, reuse):
        is_training = tf.constant(True)
        with tf.variable_scope('d4', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 128)
        return x

    def discriminator5(self, x, reuse):
        is_training = tf.constant(True)
        with tf.variable_scope('d5', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 128)
        return x

    def discriminator6(self, x, reuse):
        is_training = tf.constant(True)
        with tf.variable_scope('d6', reuse=reuse):
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 1024)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 512)
                #x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('fc3'):
                x = full_connection_layer(x, 128)
        return x


    def calc_d_loss(self):
        real_d1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_real, labels=tf.ones_like(self.D1_real)))
        fake_d1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_fake, labels=tf.zeros_like(self.D1_fake)))
        real_d2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_real, labels=tf.ones_like(self.D2_real)))
        fake_d2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_fake, labels=tf.zeros_like(self.D2_fake)))
        real_d3_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D3_real, labels=tf.ones_like(self.D3_real)))
        fake_d3_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D3_fake, labels=tf.zeros_like(self.D3_fake)))
        real_d4_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D4_real, labels=tf.ones_like(self.D4_real)))
        fake_d4_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D4_fake, labels=tf.zeros_like(self.D4_fake)))

        loss = real_d1_loss+fake_d1_loss+real_d2_loss+fake_d2_loss+real_d3_loss+fake_d3_loss+real_d4_loss+fake_d4_loss

        #cycle consistency----
        real_d5_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D5_real, labels=tf.ones_like(self.D5_real)))
        fake_d5_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D5_fake, labels=tf.zeros_like(self.D5_fake)))
        real_d6_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D6_real, labels=tf.ones_like(self.D6_real)))
        fake_d6_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D6_fake, labels=tf.zeros_like(self.D6_fake)))
        loss = loss +real_d5_loss+fake_d5_loss+real_d6_loss+fake_d6_loss
        #--------------------------------------
        return loss

    def calc_g_loss(self):
        g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_fake, labels=tf.ones_like(self.D1_fake)))
        g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_fake, labels=tf.ones_like(self.D2_fake)))
        g3_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D3_fake, labels=tf.ones_like(self.D3_fake)))
        g4_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D4_fake, labels=tf.ones_like(self.D4_fake)))
        l1 = 1
        L1_loss = l1 * tf.reduce_mean(tf.abs(self.gen_y - self.y))
        L1_loss_ = l1 * tf.reduce_mean(tf.abs(self.x - self.gen_x))
        l2 = 1
        MMD_loss1 = mmd_loss(self.gen_x, self.x)#l2 * self.compute_MMD_loss(self.gen_x, self.x)
        MMD_loss2 = mmd_loss(self.gen_fake_x, self.x)#l2 * self.compute_MMD_loss(self.gen_fake_x, self.x)

        loss = g1_loss + g2_loss + g3_loss + g4_loss + L1_loss + L1_loss_ + MMD_loss1 + MMD_loss2

        #cycle consistency----
        g5_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D5_fake, labels=tf.ones_like(self.D5_fake)))
        g6_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D6_fake, labels=tf.ones_like(self.D6_fake)))
        cc_loss1 = tf.reduce_mean(tf.abs(self.gen_y1 - self.y))
        cc_loss2 = tf.reduce_mean(tf.abs(self.gen_x1 - self.x))
        loss = loss + g5_loss + g6_loss + cc_loss1 + cc_loss2
        #---------------------------------------

        return loss
