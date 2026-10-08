import service from './index'

// Banner test: synthetic audience panel evaluates banner images.
export const createBannerTest = (formData) =>
  service.post('/api/banner-test', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })

export const getBannerTest = (testId) => service.get(`/api/banner-test/${testId}`)

export const listBannerTests = () => service.get('/api/banner-test/list')

export const getBannerAnswers = (testId) => service.get(`/api/banner-test/${testId}/answers`)

export const bannerImageUrl = (testId, label) =>
  `${service.defaults.baseURL || ''}/api/banner-test/${testId}/image/${label}`

export const landingScreenUrl = (testId, label, n) =>
  `${service.defaults.baseURL || ''}/api/banner-test/${testId}/screen/${label}/${n}`
