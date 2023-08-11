import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/linear_regression',
    name: 'Linear Regression Model',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "LR" */ '../views/LRView')
  },
  {
    path: '/logistic_regression',
    name: 'Logistic Regression Model',
    component: () => import(/* webpackChunkName: "LGR" */ '../views/LGRView')
  },
  {
    path: '/polynomial_regression',
    name: 'Polynomial Regression Model',
    component: () => import(/* webpackChunkName: "Poly" */ '../views/PolyView')
  },
  {
    path: '/k_means_clustering',
    name: 'K Means Clustering Model',
    component: () => import(/* webpackChunkName: "KMeans" */ '../views/KmeansView')
  },
  {
    path: '/svm',
    name: 'SVM Model',
    component: () => import(/* webpackChunkName: "SVM" */ '../views/SVMView')
  },
  {
    path: '/neural_network',
    name: 'Neural Network Model',
    component: () => import(/* webpackChunkName: "Neural" */ '../views/NeuralView')
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
