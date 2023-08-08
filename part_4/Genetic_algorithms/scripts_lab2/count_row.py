def show_graphic(self) -> None:
        print(end='')
        fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

        if self.X is None and self.Y is None:
            self.X, self.Y = np.meshgrid(np.arange(*self.shape_graph), np.arange(*self.shape_graph))
            self.Z = self.target_func([self.X, self.Y])

        ax1.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax2.plot_wireframe(self.X, self.Y, self.Z, rstride=10, cstride=10)

        ax1.scatter( self.population[:,0], self.population[:,1], self.population[:,2],color='red')
        ax2.scatter( self.population[:,0], self.population[:,1], self.population[:,2],color='black')
        # plt.tight_layout()
        plt.show()